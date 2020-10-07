#include <functional>

#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <math.h>
#include <eigen3/Eigen/Dense>

#include "common/plugin.hpp"
#include "common/switchboard.hpp"
#include "common/data_format.hpp"
#include "common/phonebook.hpp"

using namespace ILLIXR;

class kimera_vio : public plugin {
public:
	/* Provide handles to kimera_vio */
	kimera_vio(std::string name_, phonebook* pb_)
		: plugin{name_, pb_}
		, sb{pb->lookup_impl<switchboard>()}
		, kimera_current_frame_id(0)
		, kimera_pipeline_params("../params/ILLIXR")
		, kimera_pipeline(kimera_pipeline_params)
	{
		_m_pose = sb->publish<pose_type>("slow_pose");
		_m_imu_raw = sb->publish<imu_raw_type>("imu_raw");
		_m_begin = std::chrono::system_clock::now();
		imu_cam_buffer = NULL;

		_m_pose->put(
			new pose_type{
				.sensor_time = std::chrono::time_point<std::chrono::system_clock>{},
				.position = Eigen::Vector3f{0, 0, 0},
				.orientation = Eigen::Quaternionf{1, 0, 0, 0}
			}
		);

#ifdef CV_HAS_METRICS
		cv::metrics::setAccount(new std::string{"-1"});
#endif

	}


	virtual void start() override {
		plugin::start();
		sb->schedule<imu_cam_type>(id, "imu_cam", [&](const imu_cam_type *datum) {
			this->feed_imu_cam(datum);
		});
	}


	std::size_t iteration_no = 0;
	void feed_imu_cam(const imu_cam_type *datum) {
		// Ensures that slam doesnt start before valid IMU readings come in
		if (datum == NULL) {
			assert(previous_timestamp == 0);
			return;
		}

		// This ensures that every data point is coming in chronological order If youre failing this assert, 
		// make sure that your data folder matches the name in offline_imu_cam/plugin.cc
		assert(datum->dataset_time > previous_timestamp);
		previous_timestamp = datum->dataset_time;
		VIO::Vector6 imu_raw_vals;
		imu_raw_vals << datum->la, datum->angular_v;

		// Feed the IMU measurement. There should always be IMU data in each call to feed_imu_cam
		assert((datum->img0.has_value() && datum->img1.has_value()) || (!datum->img0.has_value() && !datum->img1.has_value()));

		kimera_pipeline->fillSingleImuQueue(VIO::ImuMeasurement(datum->dataset_time, imu_raw_vals));
		if (open_vins_estimator.initialized()) {
			Eigen::Matrix<double,13,1> state_plus = Eigen::Matrix<double,13,1>::Zero();
			imu_raw_type *imu_raw_data = new imu_raw_type {
				Eigen::Matrix<double, 3, 1>::Zero(), 
				Eigen::Matrix<double, 3, 1>::Zero(), 
				Eigen::Matrix<double, 3, 1>::Zero(), 
				Eigen::Matrix<double, 3, 1>::Zero(),
				Eigen::Matrix<double, 13, 1>::Zero(),
				// Record the timestamp (in ILLIXR time) associated with this imu sample.
				// Used for MTP calculations.
				datum->time
			};
			// TODO: Remove this and figure out where to get IMU biases from kimera.
        	// open_vins_estimator.get_propagator()->fast_state_propagate(state, timestamp_in_seconds, state_plus, imu_raw_data);

			_m_imu_raw->put(imu_raw_data);
		}

		// If there is not cam data this func call, break early
		if (!datum->img0.has_value() && !datum->img1.has_value()) {
			kimera_pipeline.spin();
			return;
		} else if (imu_cam_buffer == NULL) {
			imu_cam_buffer = datum;
			kimera_pipeline.spin();
			return;
		}

#ifdef CV_HAS_METRICS
		cv::metrics::setAccount(new std::string{std::to_string(iteration_no)});
		iteration_no++;
		if (iteration_no % 20 == 0) {
			cv::metrics::dump();
		}
#else
#warning "No OpenCV metrics available. Please recompile OpenCV from git clone --branch 3.4.6-instrumented https://github.com/ILLIXR/opencv/. (see install_deps.sh)"
#endif

		cv::Mat img0{*imu_cam_buffer->img0.value()};
		cv::Mat img1{*imu_cam_buffer->img1.value()};
		cv::cvtColor(img0, img0, cv::COLOR_BGR2GRAY);
		cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
		double buffer_timestamp_seconds = double(imu_cam_buffer->dataset_time) / NANO_SEC;
		// VIOParams
		VIO::CameraParams left_cam_info = kimera_pipeline_params.camera_params_.at(0);
		VIO::CameraParams right_cam_info = kimera_pipeline_params.camera_params_.at(1);
		kimera_pipeline->fillLeftFrameQueue(VIO::make_unique<Frame>(kimera_current_frame_id_,
																	datum->dataset_time,
																	left_cam_info, img0));
		kimera_pipeline->fillRightFrameQueue(VIO::make_unique<Frame>(kimera_current_frame_id_,
																	 datum->dataset_time,
																	 right_cam_info, img1));

		kimera_pipeline.spin();

		// Get the pose returned from SLAM
		state = open_vins_estimator.get_state();
		Eigen::Vector4d quat = state->_imu->quat();
		Eigen::Vector3d pose = state->_imu->pos();

		Eigen::Vector3f swapped_pos = Eigen::Vector3f{float(pose(0)), float(pose(1)), float(pose(2))};
		Eigen::Quaternionf swapped_rot = Eigen::Quaternionf{float(quat(3)), float(quat(0)), float(quat(1)), float(quat(2))};

       	assert(isfinite(swapped_rot.w()));
        assert(isfinite(swapped_rot.x()));
        assert(isfinite(swapped_rot.y()));
        assert(isfinite(swapped_rot.z()));
        assert(isfinite(swapped_pos[0]));
        assert(isfinite(swapped_pos[1]));
        assert(isfinite(swapped_pos[2]));

		if (open_vins_estimator.initialized()) {
			if (isUninitialized) {
				isUninitialized = false;
			}

			_m_pose->put(new pose_type{
				.sensor_time = imu_cam_buffer->time,
				.position = swapped_pos,
				.orientation = swapped_rot,
			});
		}

		// I know, a priori, nobody other plugins subscribe to this topic
		// Therefore, I can const the cast away, and delete stuff
		// This fixes a memory leak.
		// -- Sam at time t1
		// Turns out, this is no longer correct. debbugview uses it
		// const_cast<imu_cam_type*>(imu_cam_buffer)->img0.reset();
		// const_cast<imu_cam_type*>(imu_cam_buffer)->img1.reset();
		imu_cam_buffer = datum;
	}

	virtual ~kimera_vio() override {}

private:
	const std::shared_ptr<switchboard> sb;
	std::unique_ptr<writer<pose_type>> _m_pose;
	std::unique_ptr<writer<imu_raw_type>> _m_imu_raw;
	time_type _m_begin;
	State *state;

	VIO::FrameId kimera_current_frame_id;
	VIO::VioParams kimera_pipeline_params;
	VIO::Pipeline kimera_pipeline;
	
	const imu_cam_type* imu_cam_buffer;
	double previous_timestamp = 0.0;
	bool isUninitialized = true;
};

PLUGIN_MAIN(kimera_vio)

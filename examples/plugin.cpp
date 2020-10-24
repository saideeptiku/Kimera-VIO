#include <functional>

#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <math.h>
#include <eigen3/Eigen/Dense>

#include "kimera-vio/pipeline/Pipeline.h"

#include "../common/plugin.hpp"
#include "../common/switchboard.hpp"
#include "../common/data_format.hpp"
#include "../common/phonebook.hpp"

using namespace ILLIXR;

class kimera_vio : public plugin {
public:
	/* Provide handles to kimera_vio */
	kimera_vio(std::string name_, phonebook* pb_)
		: plugin{name_, pb_}
		, sb{pb->lookup_impl<switchboard>()}
		, kimera_current_frame_id(0)
		// TODO: get path from runner
		, kimera_pipeline_params("/home/jeffrey/Research/ILLIXR/Kimera-VIO/params/ILLIXR")
		, kimera_pipeline(kimera_pipeline_params)
    	, _m_pose{sb->publish<pose_type>("slow_pose")}
    	, _m_imu_integrator_input{sb->publish<imu_integrator_input>("imu_integrator_input")}
	{
		_m_begin = std::chrono::system_clock::now();
		imu_cam_buffer = NULL;
    
    // TODO: read flag file path from runner and find a better way of passing it to gflag
    //int argc = 2;
    //char **argv;
    //char *argv0 = "IGNORE";
    //char *argv1 = "--flagfile=/home/huzaifa2/all_scratch/components/KV-ILLIXR/params/ILLIXR/flags/stereoVIOEuroc.flags";
    //argv[0] = argv0;
    //argv[1] = argv1;
    //char *argv[] = {"IGNORE", "--flagfile=/home/huzaifa2/all_scratch/components/KV-ILLIXR/params/ILLIXR/flags/stereoVIOEuroc.flags"};
    //google::ParseCommandLineFlags(&argc, &argv, true);

    kimera_pipeline.registerBackendOutputCallback(
      std::bind(
        &kimera_vio::pose_callback,
        this,
        std::placeholders::_1
      )
    );

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

		imu_cam_buffer = datum;

		VIO::Vector6 imu_raw_vals;
		imu_raw_vals << datum->linear_a.cast<double>(), datum->angular_v.cast<double>();

		// Feed the IMU measurement. There should always be IMU data in each call to feed_imu_cam
		assert((datum->img0.has_value() && datum->img1.has_value()) || (!datum->img0.has_value() && !datum->img1.has_value()));
		kimera_pipeline.fillSingleImuQueue(VIO::ImuMeasurement(datum->dataset_time, imu_raw_vals));

		// If there is not cam data this func call, break early
		if (!datum->img0.has_value() && !datum->img1.has_value()) {
			//kimera_pipeline.spin();
      //std::cout << "SPIN IMU\n";
			return;
		}

#ifdef CV_HAS_METRICS
		cv::metrics::setAccount(new std::string{std::to_string(itew_pose_blkf_rot
		}
#else
#warning "No OpenCV metrics available. Please recompile OpenCV from git clone --branch 3.4.6-instrumented https://github.com/ILLIXR/opencv/. (see install_deps.sh)"
#endif

		cv::Mat img0{*imu_cam_buffer->img0.value()};
		cv::Mat img1{*imu_cam_buffer->img1.value()};
		cv::cvtColor(img0, img0, cv::COLOR_BGR2GRAY);
		cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
		// VIOParams
		VIO::CameraParams left_cam_info = kimera_pipeline_params.camera_params_.at(0);
		VIO::CameraParams right_cam_info = kimera_pipeline_params.camera_params_.at(1);
		kimera_pipeline.fillLeftFrameQueue(VIO::make_unique<VIO::Frame>(kimera_current_frame_id,
																	datum->dataset_time,
																	left_cam_info, img0));
		kimera_pipeline.fillRightFrameQueue(VIO::make_unique<VIO::Frame>(kimera_current_frame_id,
																	datum->dataset_time,
																	right_cam_info, img1));

		kimera_pipeline.spin();
    std::cout << "SPIN FULL\n";
	}

  	void pose_callback(const std::shared_ptr<VIO::BackendOutput>& vio_output) {
		// std::cout << "We're in business!" << std::endl;
		// std::cout << "########################################################################\n";

		const auto& cached_state = vio_output->W_State_Blkf_;
		const auto& w_pose_blkf_trans = cached_state.pose_.translation().transpose();
		const auto& w_pose_blkf_rot = cached_state.pose_.rotation().quaternion();
		const auto& w_vel_blkf = cached_state.velocity_.transpose();
		const auto& imu_bias_gyro = cached_state.imu_bias_.gyroscope().transpose();
		const auto& imu_bias_acc = cached_state.imu_bias_.accelerometer().transpose();
		// std::cout     << cached_state.timestamp_ << ","  //
		//               << "\n"
		//               << "px = " << w_pose_blkf_trans.x() << ","    //
		//               << "py = " << w_pose_blkf_trans.y() << ","    //
		//               << "pz = " << w_pose_blkf_trans.z() << ","    //
		//               << "\n"
		//               << "qw = " << w_pose_blkf_rot(0) << ","       // q_w
		//               << "qx = " << w_pose_blkf_rot(1) << ","       // q_x
		//               << "qy = " << w_pose_blkf_rot(2) << ","       // q_y
		//               << "qz = " << w_pose_blkf_rot(3) << ","       // q_z
		//               << "\n"
		//               << w_vel_blkf(0) << ","            //
		//               << w_vel_blkf(1) << ","            //
		//               << w_vel_blkf(2) << ","            //
		//               << "\n"
		//               << "bgx = " << imu_bias_gyro(0) << "\n"         //
		//               << "bgy = " << imu_bias_gyro(1) << "\n"         //
		//               << "bgz = " << imu_bias_gyro(2) << "\n"         //
		//               << "\n"
		//               << "bax = " << imu_bias_acc(0) << "\n"          //
		//               << "bay = " << imu_bias_acc(1) << "\n"          //
		//               << "baz = " << imu_bias_acc(2) << "\n"          //
		//               << std::endl;

		// Get the pose returned from SLAM
		Eigen::Quaternionf quat = Eigen::Quaternionf{w_pose_blkf_rot(0), w_pose_blkf_rot(1), w_pose_blkf_rot(2), w_pose_blkf_rot(3)};
		Eigen::Quaterniond doub_quat = Eigen::Quaterniond{w_pose_blkf_rot(0), w_pose_blkf_rot(1), w_pose_blkf_rot(2), w_pose_blkf_rot(3)};
		Eigen::Vector3f pos  = w_pose_blkf_trans.cast<float>();

		assert(isfinite(quat.w()));
		assert(isfinite(quat.x()));
		assert(isfinite(quat.y()));
		assert(isfinite(quat.z()));
		assert(isfinite(pos[0]));
		assert(isfinite(pos[1]));
		assert(isfinite(pos[2]));

		_m_pose->put(new pose_type{
		.sensor_time = imu_cam_buffer->time,
		.position = pos,
		.orientation = quat,
		});

		_m_imu_integrator_input->put(new imu_integrator_input{
			.last_cam_integration_time = (double(imu_cam_buffer->dataset_time) / NANO_SEC),
			.t_offset = -0.05,

			.params = {
				.gyro_noise = 0.00016968,
				.acc_noise = 0.002,
				.gyro_walk = 1.9393e-05,
				.acc_walk = 0.003,
				.n_gravity = Eigen::Matrix<double,3,1>(0.0, 0.0, -9.81),
				.imu_integration_sigma = 1.0,
				.nominal_rate = 200.0,
			},

			.biasAcc =imu_bias_acc,
			.biasGyro = imu_bias_gyro,
			.position = w_pose_blkf_trans,
			.velocity = w_vel_blkf,
			.quat = doub_quat,
		});
	}

	virtual ~kimera_vio() override {}

private:
	const std::shared_ptr<switchboard> sb;
	std::unique_ptr<writer<pose_type>> _m_pose;
	std::unique_ptr<writer<imu_integrator_input>> _m_imu_integrator_input;
	//std::unique_ptr<writer<imu_integrator_input>> _m_imu_integrator_input;
	time_type _m_begin;

	VIO::FrameId kimera_current_frame_id;
	VIO::VioParams kimera_pipeline_params;
	VIO::Pipeline kimera_pipeline;
	
	const imu_cam_type* imu_cam_buffer;
	double previous_timestamp = 0.0;
};

PLUGIN_MAIN(kimera_vio)

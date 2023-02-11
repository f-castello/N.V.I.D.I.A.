// STL
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace std;
#include <chrono>
using namespace std::chrono;

// OpenCV
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
using namespace cv;
#include <opencv2/core/utils/filesystem.hpp>
using namespace cv::utils::fs;
#include <opencv2/core/cuda.hpp>
using namespace cv::cuda;

// prototypes
void deblock1h(uint8_t *data, uint8_t QF, size_t width, size_t height,
							 size_t idx);
void deblock1v(uint8_t *data, uint8_t QF, size_t width, size_t idx);
__global__ void gpudeblock1h(uint8_t *data, uint8_t QF, size_t width,
														 size_t height, size_t step);
__global__ void gpudeblock1v(uint8_t *data, uint8_t QF, size_t width,
														 size_t height, size_t step);

// N.V.I.D.I.A.
int main(int argc, char *argv[]) {

	// declaration of non-const variables/objects
	string input, output, extension, second;
	bool specified;
	string::size_type sz;
	int factor;
	size_t number = 0;
	steady_clock::time_point start, horizontal, vertical;
	duration<double, milli> deltah, deltav, partial;
	duration<double> totalh = steady_clock::duration::zero();
	duration<double> totalv = steady_clock::duration::zero();
	duration<double> final = steady_clock::duration::zero();
	Mat frame, channel[3];
	GpuMat gpuchannel[3];
	Stream stream[3];
	Event upload0[3], upload1[3], gpuhorizontal0[3], gpuhorizontal1[3],
			gpuvertical0[3], gpuvertical1[3], download0[3], download1[3];
	float gpudeltaup, transferpartial, gpudeltah, gpudeltav, kernelpartial,
			gpudeltadown, gpupartial;
	float gpufinal = 0, totaltransfer = 0, totalup = 0, totaldown = 0,
				totalkernel = 0, gputotalh = 0, gputotalv = 0;

	// argument(s) parsing
	switch (argc) {
	case 1:
		cerr << "Error: unspecified input video file." << endl;
		exit(EXIT_FAILURE);
	case 2:
		clog << "Note: unspecified quantization factor, defaulting to 127." << endl;
		factor = 127;
		clog << "Note: unspecified output video title, defaulting to same as input."
				 << endl;
		output = argv[1];
		specified = false;
		break;
	default:
		cerr << "Warning: ignoring redundant argument(s)." << endl;
	case 4:
		second = argv[2];
		try {
			factor = stoi(second, &sz);
			if (sz < second.length()) {
				cerr << "Warning: ignoring redundant character(s) from quantization "
								"factor."
						 << endl;
			}
			if (factor < 1 || factor > 255) {
				throw out_of_range("out_of_range");
			}
		} catch (invalid_argument &e) {
			cerr << "Error: invalid quantization factor is not a number [1-255]."
					 << endl;
			exit(EXIT_FAILURE);
		} catch (out_of_range &e) {
			cerr << "Error: out of range quantization factor [1-255]." << endl;
			exit(EXIT_FAILURE);
		}
		output = argv[3];
		specified = true;
		break;
	case 3:
		second = argv[2];
		try {
			factor = stoi(second, &sz);
			if (sz < second.length()) {
				cerr << "Warning: ignoring redundant character(s) from quantization "
								"factor."
						 << endl;
			}
			if (factor < 1 || factor > 255) {
				throw out_of_range("out_of_range");
			}
			clog << "Note: unspecified output video title, defaulting to same as "
							"input."
					 << endl;
			output = argv[1];
			specified = false;
		} catch (invalid_argument &e) {
			output = argv[2];
			specified = true;
			clog << "Note: unspecified quantization factor, defaulting to 127."
					 << endl;
			factor = 127;
		} catch (out_of_range &e) {
			cerr << "Error: out of range quantization factor [1-255]." << endl;
			exit(EXIT_FAILURE);
		}
		break;
	}

	// I/O filenames management
	input = argv[1];
	try {
		extension = input.substr(input.find_last_of("."));
	} catch (out_of_range &e) {
		cerr << "Warning: unspecified video file exension." << endl;
	} catch (bad_alloc &e) {
		cerr << "Error: insufficient memory." << endl;
		exit(EXIT_FAILURE);
	}
	string::size_type const sep(output.find_last_of("/\\"));
	if (output.find_last_of("/\\") != string::npos && specified) {
		cerr << "Warning: ignoring unexpected file path in output video title."
				 << endl;
	}
	output = output.substr(sep + 1);
	string::size_type const pos(output.find_last_of("."));
	if ((pos) > 0 && (pos != string::npos)) {
		if (specified) {
			cerr << "Warning: ignoring unexpected file extension in output video "
							"title."
					 << endl;
		}
		output = output.substr(0, pos);
	}
	output += extension;

	// input capture check
	VideoCapture capture(input);
	if (!capture.isOpened()) {
		cerr << "Error: no such file or directory (" << input << ")." << endl;
		exit(EXIT_FAILURE);
	}
	const size_t width = capture.get(CAP_PROP_FRAME_WIDTH);
	const size_t height = capture.get(CAP_PROP_FRAME_HEIGHT);

	// output writers check
	createDirectory("cpu");
	VideoWriter writer("./cpu/" + output, capture.get(CAP_PROP_FOURCC),
										 capture.get(CAP_PROP_FPS), Size(width, height));
	createDirectory("gpu");
	VideoWriter gpuwriter("./gpu/" + output, capture.get(CAP_PROP_FOURCC),
												capture.get(CAP_PROP_FPS), Size(width, height));
	if (!writer.isOpened()) {
		cerr << "Error: permission denied (./cpu/" << output << ")." << endl;
		exit(EXIT_FAILURE);
	}
	if (!gpuwriter.isOpened()) {
		cerr << "Error: permission denied (./gpu/" << output << ")." << endl;
		exit(EXIT_FAILURE);
	}

	// scan video frame by frame
	while (1) {
		capture >> frame;
		if (frame.empty()) {
			break;
		}
		cout << fixed;
		cout.precision(0);
		cout << "\nProcessing frame " << ++number << "/"
				 << capture.get(CAP_PROP_FRAME_COUNT) << "..." << endl;

		// convert to YCrCb color space and separate luma + chroma channels
		cvtColor(frame, frame, COLOR_BGR2YCrCb);
		split(frame, channel);

		// reset atomic GPU timings
		gpudeltaup = 0;
		gpudeltadown = 0;
		gpudeltah = 0;
		gpudeltav = 0;

		// launch memory transfer(s) from host to device
		for (size_t ch = 0; ch < 3; ++ch) {
			upload0[ch].record(stream[ch]);
			gpuchannel[ch].upload(channel[ch], stream[ch]);
			upload1[ch].record(stream[ch]);
		}
		for (size_t ch = 0; ch < 3; ++ch) {
			upload1[ch].waitForCompletion();
			gpudeltaup += Event::elapsedTime(upload0[ch], upload1[ch]);
		}

		// host will work on each pixel of each component separately
		deltah = steady_clock::duration::zero();
		deltav = steady_clock::duration::zero();
		for (size_t ch = 0; ch < 3; ++ch) {
			start = steady_clock::now();
			for (size_t p = 0; p < width * height; ++p) {
				deblock1h((uint8_t *)channel[ch].data, factor, width, height, p);
			}
			horizontal = steady_clock::now();
			deltah += duration_cast<milliseconds>(horizontal - start);
			for (size_t p = 0; p < width * height; ++p) {
				deblock1v((uint8_t *)channel[ch].data, factor, width, p);
			}
			vertical = steady_clock::now();
			deltav += duration_cast<milliseconds>(vertical - horizontal);
		}
		partial = deltah + deltav;
		totalh += deltah;
		totalv += deltav;
		final += partial;
		cout << "- CPU frame time: " << partial.count() << " ms" << endl;
		cout << "  - Horizontal pass: " << deltah.count() << " ms" << endl;
		cout << "  - Vertical pass: " << deltav.count() << " ms" << endl;

		// reconstruct & convert new frame for the CPU output
		merge(channel, 3, frame);
		cvtColor(frame, frame, COLOR_YCrCb2BGR);
		writer.write(frame);

		// device will run a stream for each channel and infer implicit loops
		const dim3 gridSize(ceil((float)width / 32), ceil((float)height / 32));
		const dim3 blockSize(32, 32);
		for (size_t ch = 0; ch < 3; ++ch) {
			gpuhorizontal0[ch].record(stream[ch]);
			gpudeblock1h<<<gridSize, blockSize, 0,
										 (CUstream_st *)stream[ch].cudaPtr()>>>(
					(uint8_t *)gpuchannel[ch].data, factor, width, height,
					gpuchannel[ch].step);
			gpuhorizontal1[ch].record(stream[ch]);
		}
		for (size_t ch = 0; ch < 3; ++ch) {
			gpuvertical0[ch].record(stream[ch]);
			gpudeblock1v<<<gridSize, blockSize, 0,
										 (CUstream_st *)stream[ch].cudaPtr()>>>(
					(uint8_t *)gpuchannel[ch].data, factor, width, height,
					gpuchannel[ch].step);
			gpuvertical1[ch].record(stream[ch]);
		}
		for (size_t ch = 0; ch < 3; ++ch) {
			gpuhorizontal1[ch].waitForCompletion();
			gpudeltah += Event::elapsedTime(gpuhorizontal0[ch], gpuhorizontal1[ch]);
		}
		for (size_t ch = 0; ch < 3; ++ch) {
			gpuvertical1[ch].waitForCompletion();
			gpudeltav += Event::elapsedTime(gpuvertical0[ch], gpuvertical1[ch]);
		}

		// launch memory transfer(s) from device to host
		for (size_t ch = 0; ch < 3; ++ch) {
			download0[ch].record(stream[ch]);
			gpuchannel[ch].download(channel[ch], stream[ch]);
			download1[ch].record(stream[ch]);
		}
		for (size_t ch = 0; ch < 3; ++ch) {
			download1[ch].waitForCompletion();
			gpudeltadown += Event::elapsedTime(download0[ch], download1[ch]);
		}

		// set partial GPU timings
		transferpartial = gpudeltaup + gpudeltadown;
		kernelpartial = gpudeltah + gpudeltav;
		gpupartial = transferpartial + kernelpartial;

		// update cumulative GPU timings
		totalup += gpudeltaup;
		totaldown += gpudeltadown;
		gputotalh += gpudeltah;
		gputotalv += gpudeltav;

		// reconstruct & convert new frame for the GPU output
		cout << "- GPU frame time: " << gpupartial << " ms" << endl;
		cout << "  - Data transfers: " << transferpartial << " ms" << endl;
		cout << "    - HostToDevice: " << gpudeltaup << " ms" << endl;
		cout << "    - DeviceToHost: " << gpudeltadown << " ms" << endl;
		cout << "  - Kernel calls: " << kernelpartial << " ms" << endl;
		cout << "    - Horizontal pass: " << gpudeltah << " ms" << endl;
		cout << "    - Vertical pass: " << gpudeltav << " ms" << endl;
		merge(channel, 3, frame);
		cvtColor(frame, frame, COLOR_YCrCb2BGR);
		gpuwriter.write(frame);
	}

	// update final GPU timings
	totaltransfer = totalup + totaldown;
	totalkernel = gputotalh + gputotalv;
	gpufinal = totaltransfer + totalkernel;
	cout << "\nVideo processing complete, please wait..." << endl;

	// final comparisons and program termination
	writer.release();
	capture.release();
	gpuwriter.release();
	cout.precision(2);
	cout << "- Cumulative CPU time: " << final.count() << " s" << endl;
	cout << "  - Horizontal pass: " << totalh.count() << " s" << endl;
	cout << "  - Vertical pass: " << totalv.count() << " s" << endl;
	cout << "- Cumulative GPU time: " << gpufinal / 1000 << " s" << endl;
	cout << "  - Data transfers: " << totaltransfer / 1000 << " s" << endl;
	cout << "    - HostToDevice: " << totalup / 1000 << " s" << endl;
	cout << "    - DeviceToHost: " << totaldown / 1000 << " s" << endl;
	cout << "  - Kernel calls: " << totalkernel / 1000 << " s" << endl;
	cout << "    - Horizontal pass: " << gputotalh / 1000 << " s" << endl;
	cout << "    - Vertical pass: " << gputotalv / 1000 << " s\n" << endl;
	return 0;
}

// algorithm implementation(s) with saturation towards 0 (min) or 255 (max)
void deblock1h(uint8_t *data, uint8_t QF, size_t width, size_t height,
							 size_t idx) {

	// determine position
	const size_t row = idx / width;

	// horizontal edge filtering
	if ((row) && (row < height - 1) && !(row % 8)) {
		uint8_t B = data[idx - 2 * width];
		uint8_t C = data[idx - width];
		uint8_t D = data[idx];
		uint8_t E = data[idx + width];
		if (abs(B - C) < 5 && abs(D - E) < 5) {

			// strong filtering
			if (row < height - 2) {
				uint8_t A = data[idx - 3 * width];
				uint8_t F = data[idx + 2 * width];
				if (abs(C - D) < 2.0 * QF) {
					int16_t x = (int16_t)(D - C);
					int16_t a = (int16_t)(A + (x >> 3));
					data[idx - 3 * width] = -(a >> 8) | (uint8_t)a;
					int16_t b = (int16_t)(B + (x >> 2));
					data[idx - 2 * width] = -(b >> 8) | (uint8_t)b;
					int16_t c = (int16_t)(C + (x >> 1));
					data[idx - width] = -(c >> 8) | (uint8_t)c;
					int16_t d = (int16_t)(D - (x >> 1));
					data[idx] = (~d >> 8) & (uint8_t)d;
					int16_t e = (int16_t)(E - (x >> 2));
					data[idx + width] = (~e >> 8) & (uint8_t)e;
					int16_t f = (int16_t)(F - (x >> 3));
					data[idx + 2 * width] = (~f >> 8) & (uint8_t)f;
				}
			}
		} else {

			// weak filtering
			if (abs(C - D) < 0.8 * QF) {
				int16_t x = (int16_t)(D - C);
				int16_t b = (int16_t)(B + (x >> 3));
				data[idx - 2 * width] = -(b >> 8) | (uint8_t)b;
				int16_t c = (int16_t)(C + (x >> 1));
				data[idx - width] = -(c >> 8) | (uint8_t)c;
				int16_t d = (int16_t)(D - (x >> 1));
				data[idx] = (~d >> 8) & (uint8_t)d;
				int16_t e = (int16_t)(E - (x >> 3));
				data[idx + width] = (~e >> 8) & (uint8_t)e;
			}
		}
	}
}
void deblock1v(uint8_t *data, uint8_t QF, size_t width, size_t idx) {

	// determine position
	const size_t col = idx % width;

	// vertical edge filtering
	if ((col) && (col < width - 1) && !(col % 8)) {
		uint8_t B = data[idx - 2];
		uint8_t C = data[idx - 1];
		uint8_t D = data[idx];
		uint8_t E = data[idx + 1];
		if (abs(B - C) < 5 && abs(D - E) < 5) {

			// strong filtering
			if (col < width - 2) {
				uint8_t A = data[idx - 3];
				uint8_t F = data[idx + 2];
				if (abs(C - D) < 2.0 * QF) {
					int16_t x = (int16_t)(D - C);
					int16_t a = (int16_t)(A + (x >> 3));
					data[idx - 3] = -(a >> 8) | (uint8_t)a;
					int16_t b = (int16_t)(B + (x >> 2));
					data[idx - 2] = -(b >> 8) | (uint8_t)b;
					int16_t c = (int16_t)(C + (x >> 1));
					data[idx - 1] = -(c >> 8) | (uint8_t)c;
					int16_t d = (int16_t)(D - (x >> 1));
					data[idx] = (~d >> 8) & (uint8_t)d;
					int16_t e = (int16_t)(E - (x >> 2));
					data[idx + 1] = (~e >> 8) & (uint8_t)e;
					int16_t f = (int16_t)(F - (x >> 3));
					data[idx + 2] = (~f >> 8) & (uint8_t)f;
				}
			}
		} else {

			// weak filtering
			if (abs(C - D) < 0.8 * QF) {
				int16_t x = (int16_t)(D - C);
				int16_t b = (int16_t)(B + (x >> 3));
				data[idx - 2] = -(b >> 8) | (uint8_t)b;
				int16_t c = (int16_t)(C + (x >> 1));
				data[idx - 1] = -(c >> 8) | (uint8_t)c;
				int16_t d = (int16_t)(D - (x >> 1));
				data[idx] = (~d >> 8) & (uint8_t)d;
				int16_t e = (int16_t)(E - (x >> 3));
				data[idx + 1] = (~e >> 8) & (uint8_t)e;
			}
		}
	}
}
__global__ void gpudeblock1h(uint8_t *data, uint8_t QF, size_t width,
														 size_t height, size_t step) {

	// determine position
	size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	size_t row = threadIdx.y + blockIdx.y * blockDim.y;

	// horizontal edge filtering
	if ((col < width) && (row) && (row < height - 1) && !(row % 8)) {
		size_t idx = row * step + col;
		uint8_t B = data[idx - 2 * step];
		uint8_t C = data[idx - step];
		uint8_t D = data[idx];
		uint8_t E = data[idx + step];
		if (abs(B - C) < 5 && abs(D - E) < 5) {

			// strong filtering
			if (row < height - 2) {
				uint8_t A = data[idx - 3 * step];
				uint8_t F = data[idx + 2 * step];
				if (abs(C - D) < 2.0 * QF) {
					int16_t x = (int16_t)(D - C);
					int16_t a = (int16_t)(A + (x >> 3));
					data[idx - 3 * step] = -(a >> 8) | (uint8_t)a;
					int16_t b = (int16_t)(B + (x >> 2));
					data[idx - 2 * step] = -(b >> 8) | (uint8_t)b;
					int16_t c = (int16_t)(C + (x >> 1));
					data[idx - step] = -(c >> 8) | (uint8_t)c;
					int16_t d = (int16_t)(D - (x >> 1));
					data[idx] = (~d >> 8) & (uint8_t)d;
					int16_t e = (int16_t)(E - (x >> 2));
					data[idx + step] = (~e >> 8) & (uint8_t)e;
					int16_t f = (int16_t)(F - (x >> 3));
					data[idx + 2 * step] = (~f >> 8) & (uint8_t)f;
				}
			}
		} else {

			// weak filtering
			if (abs(C - D) < 0.8 * QF) {
				int16_t x = (int16_t)(D - C);
				int16_t b = (int16_t)(B + (x >> 3));
				data[idx - 2 * step] = -(b >> 8) | (uint8_t)b;
				int16_t c = (int16_t)(C + (x >> 1));
				data[idx - step] = -(c >> 8) | (uint8_t)c;
				int16_t d = (int16_t)(D - (x >> 1));
				data[idx] = (~d >> 8) & (uint8_t)d;
				int16_t e = (int16_t)(E - (x >> 3));
				data[idx + step] = (~e >> 8) & (uint8_t)e;
			}
		}
	}
}
__global__ void gpudeblock1v(uint8_t *data, uint8_t QF, size_t width,
														 size_t height, size_t step) {

	// determine position
	size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	size_t row = threadIdx.y + blockIdx.y * blockDim.y;

	// vertical edge filtering
	if ((col) && (col < width - 1) && !(col % 8) && (row < height)) {
		size_t idx = row * step + col;
		uint8_t B = data[idx - 2];
		uint8_t C = data[idx - 1];
		uint8_t D = data[idx];
		uint8_t E = data[idx + 1];
		if (abs(B - C) < 5 && abs(D - E) < 5) {

			// strong filtering
			if (col < width - 2) {
				uint8_t A = data[idx - 3];
				uint8_t F = data[idx + 2];
				if (abs(C - D) < 2.0 * QF) {
					int16_t x = (int16_t)(D - C);
					int16_t a = (int16_t)(A + (x >> 3));
					data[idx - 3] = -(a >> 8) | (uint8_t)a;
					int16_t b = (int16_t)(B + (x >> 2));
					data[idx - 2] = -(b >> 8) | (uint8_t)b;
					int16_t c = (int16_t)(C + (x >> 1));
					data[idx - 1] = -(c >> 8) | (uint8_t)c;
					int16_t d = (int16_t)(D - (x >> 1));
					data[idx] = (~d >> 8) & (uint8_t)d;
					int16_t e = (int16_t)(E - (x >> 2));
					data[idx + 1] = (~e >> 8) & (uint8_t)e;
					int16_t f = (int16_t)(F - (x >> 3));
					data[idx + 2] = (~f >> 8) & (uint8_t)f;
				}
			}
		} else {

			// weak filtering
			if (abs(C - D) < 0.8 * QF) {
				int16_t x = (int16_t)(D - C);
				int16_t b = (int16_t)(B + (x >> 3));
				data[idx - 2] = -(b >> 8) | (uint8_t)b;
				int16_t c = (int16_t)(C + (x >> 1));
				data[idx - 1] = -(c >> 8) | (uint8_t)c;
				int16_t d = (int16_t)(D - (x >> 1));
				data[idx] = (~d >> 8) & (uint8_t)d;
				int16_t e = (int16_t)(E - (x >> 3));
				data[idx + 1] = (~e >> 8) & (uint8_t)e;
			}
		}
	}
}

// special thanks to my "debugging duck" Sara

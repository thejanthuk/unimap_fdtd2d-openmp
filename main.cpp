#include <chrono>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifdef _OPENMP
	#include <omp.h>
#endif

using real = double;

constexpr real pi = 3.14159265358979323846L;

static double now()
{
	return std::chrono::duration<double>(
		     std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void write_csv(const char *path, const real *ez,
											const real *hx, const real *hy,
										  int point_n, real dx, real dy)
{
	FILE *f = std::fopen(path, "w");
	if (!f) {
		std::perror(path);
		return;
	}

	std::fprintf(f, "x,y,z,hx,hy,ez\n");

	const int last = point_n - 1;
	for (int j = 0; j < point_n; j++) {
		for (int i = 0; i < point_n; i++) {
			const int k = i * point_n + j;

			const real hx_n = (j == 0)    ? hx[k + 1]
			                : (j == last) ? hx[k]
			                : 0.5 * (hx[k] + hx[k + 1]);
			const real hy_n = (i == 0)    ? hy[k + point_n]
			                : (i == last) ? hy[k]
			                : 0.5 * (hy[k] + hy[k + point_n]);

			std::fprintf(f, "%.12g,%.12g,0,%.12g,%.12g,%.12g\n",
			                i * dx, j * dy, hx_n, hy_n, ez[k]);
		}
	}

	std::fclose(f);
}

int main(int argc, char **argv)
{
	const real length = 1;

	const int n = std::atoi(argv[1]);

	const int point_n = n * 2 + 1;
	const int timestep_n = std::atoi(argv[2]);

	const real stable_coeff = std::atof(argv[3]);

#ifdef _OPENMP
	omp_set_num_threads(std::atoi(argv[4]));
#endif

	const int out_freq = (n + 9) / 10;

	const real delta_x = length / (n * 2);
	const real delta_y = length / (n * 2);
	const real delta_t = stable_coeff * delta_x * delta_y
											 / std::hypot(delta_x, delta_y);

	const real cx = delta_t / delta_x;
	const real cy = delta_t / delta_y;

	const real omega = pi * std::sqrt(2);
	const real s = std::sin(omega * delta_t / 2) / std::sqrt(2);

	real *hx = (real *) std::calloc(point_n * point_n, sizeof *hx);
	real *hy = (real *) std::calloc(point_n * point_n, sizeof *hy);
	real *ez = (real *) std::calloc(point_n * point_n, sizeof *ez);

	double init_beg = now();

#pragma omp parallel
{
	#pragma omp for schedule(static) nowait
	for (int i = 0; i < point_n; i++) {
		for (int j = 1; j < point_n; j++) {
			real x = i * delta_x;
			real y = (j - 0.5) * delta_y;

			hx[i * point_n + j] =  s * std::sin(pi * x) * std::cos(pi * y);
		}
	}

	#pragma omp for schedule(static) nowait
	for (int i = 1; i < point_n; i++) {
		for (int j = 0; j < point_n; j++) {
			real x = (i - 0.5) * delta_x;
			real y = j * delta_y;

			hy[i * point_n + j] = -s * std::cos(pi * x) * std::sin(pi * y);
		}
	}

	#pragma omp for schedule(static) nowait
	for (int i = 1; i < point_n - 1; i++) {
		for (int j = 1; j < point_n - 1; j++) {
			real x = i * delta_x;
			real y = j * delta_y;

			ez[i * point_n + j] = std::sin(pi * x) * std::sin(pi * y);
		}
	}
} // #pragma omp parallel
	
	double init_end = now();

	double comp_beg = now();

	if (out_freq > 0)
		write_csv("field_0000.csv", ez, hx, hy, point_n, delta_x, delta_y);

#pragma omp parallel
{
	for (int t = 0; t < timestep_n; t++) {
		#pragma omp for schedule(static) nowait
		for (int i = 0; i < point_n; i++) {
			for (int j = 1; j < point_n; j++) {
				hx[i * point_n + j] -= cy * (ez[i * point_n + j]
																		 - ez[i * point_n + j - 1]);
			}
		}

		#pragma omp for schedule(static)
		for (int i = 1; i < point_n; i++) {
			for (int j = 0; j < point_n; j++) {
				hy[i * point_n + j] += cx * (ez[i * point_n + j]
																		 - ez[(i - 1) * point_n + j]);
			}
		}

		#pragma omp for schedule(static)
		for (int i = 1; i < point_n - 1; i++) {
			for (int j = 1; j < point_n - 1; j++) {
				ez[i * point_n + j] += cx * (hy[(i + 1) * point_n + j]
																		 - hy[i * point_n + j])
														 - cy * (hx[i * point_n + j + 1]
																     - hx[i * point_n + j]);
			}
		}

		/*
		if (out_freq > 0 && (t + 1) % out_freq == 0) {
#pragma omp single
{
			char name[64];
			std::snprintf(name, sizeof name, "field_%04d.csv",
					(t + 1) / out_freq);

			write_csv(name, ez, hx, hy, point_n, delta_x, delta_y);
} // #pragma omp single
		}
		*/
	}
} // #pragma omp parallel

	double comp_end = now();

	double init_s = init_end - init_beg;
	double comp_s = comp_end - comp_beg;

#ifdef _OPENMP
	const int threads = omp_get_max_threads();
#else
	const int threads = 1;
#endif

	std::fprintf(stderr, "[init]: %.12f\n", init_s);
	std::fprintf(stderr, "[comp]: %.12f\n", comp_s);

	const real t_end = timestep_n * delta_t;
	const real cos_wt = std::cos(omega * t_end);

	real sum_sq = 0;

  #pragma omp parallel for schedule(static) reduction(+ : sum_sq)
	for (int i = 1; i < point_n - 1; i++) {
		for (int j = 1; j < point_n - 1; j++) {
			real x = i * delta_x;
			real y = j * delta_y;

			real exact = std::sin(pi * x) * std::sin(pi * y) * cos_wt;

			real diff = ez[i * point_n + j] - exact;

			sum_sq += diff * diff;
		}
	}

	const int interior_n = (point_n - 2) * (point_n - 2);
	const real rms_error = std::sqrt(sum_sq / interior_n);

	const real ez_center = ez[n * point_n + n];
	const real exact_center = cos_wt;

	std::printf("grid         : %d x %d cells (n = %d)\n", 2 * n, 2 * n, n);
	std::printf("dx, dt       : %g, %g (Courant c*dt/dx = %g)\n", delta_x, delta_t, delta_t / delta_x);
	std::printf("time steps   : %d, t = %.6f\n", timestep_n, t_end);
	std::printf("Ez(0.5, 0.5) : %.12f (exact %.12f)\n", ez_center, exact_center);
	std::printf("RMS error    : %.12e (interior nodes)\n", rms_error);

	FILE *file = std::fopen("run.csv", "a");
	std::fseek(file, 0, SEEK_END);
	if (std::ftell(file) == 0)
		std::fprintf(file,
				"cells,steps,threads,init_s,compute_s,rms_error\n");
	std::fprintf(file, "%d,%d,%d,%.12f,%.12f,%.12e\n",
			point_n - 1, timestep_n, threads, init_s, comp_s, rms_error);
	std::fclose(file);

	/*
	write_csv("result.csv", ez, hx, hy, point_n, delta_x, delta_y);
	*/

	std::free(hx);
	std::free(hy);
	std::free(ez);

	return 0;
}

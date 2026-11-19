#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

using real = double;

constexpr real pi = 3.14159265358979323846L;

//Wall-clock time in seconds (steady_clock ticks every 0.1 us, omp_get_wtime only every 1 ms with MinGW)
double wall_time()
{
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

//Write Ez as x,y,z (z = Ez) for ParaView, overwriting the file if it exists
void write_csv(const char *filename, const real *ez, int point_n, real delta_x, real delta_y)
{
	FILE *csv = std::fopen(filename, "w");
	if (!csv)
	{
		std::fprintf(stderr, "error: cannot write %s\n", filename);
		return;
	}

	std::fprintf(csv, "x,y,z\n");
	for (int j = 0; j < point_n; j++)
	{
		for (int i = 0; i < point_n; i++)
		{
			std::fprintf(csv, "%.10g,%.10g,%.10g\n", i * delta_x, j * delta_y, ez[i * point_n + j]);
		}
	}
	std::fclose(csv);
}

int main(int argc, char **argv)
{
	const real length = 1;

	const int n = std::atoi(argv[1]);

	const int point_n = n * 2 + 1;
	const int timestep_n = std::atoi(argv[2]);

	const real stable_coeff = 1/std::sqrt(2);

	const real delta_x = length / (n * 2);
	const real delta_y = length / (n * 2);
	const real delta_t = stable_coeff * delta_x * delta_y / std::hypot(delta_x, delta_y);

	const real cx = delta_t / delta_x;
	const real cy = delta_t / delta_y;

	const real omega = pi * std::sqrt(2);
	const real s = std::sin(omega * delta_t / 2) / std::sqrt(2);

	real *hx = (real *)std::calloc(point_n * point_n, sizeof *hx);
	real *hy = (real *)std::calloc(point_n * point_n, sizeof *hy);
	real *ez = (real *)std::calloc(point_n * point_n, sizeof *ez);

	//Start timing (computation only)
	double time_start = wall_time();

	/*Initailize the array*/
#pragma omp parallel
	{
#pragma omp for schedule(static) nowait
		for (int i = 0; i < point_n; i++)
		{
			for (int j = 1; j < point_n; j++)
			{
				real x = i * delta_x;
				real y = (j - 0.5) * delta_y;

				hx[i * point_n + j] = s * std::sin(pi * x) * std::cos(pi * y);
			}
		}

#pragma omp for schedule(static) nowait
		for (int i = 1; i < point_n; i++)
		{
			for (int j = 0; j < point_n; j++)
			{
				real x = (i - 0.5) * delta_x;
				real y = j * delta_y;

				hy[i * point_n + j] = -s * std::cos(pi * x) * std::sin(pi * y);
			}
		}

#pragma omp for schedule(static) nowait
		for (int i = 1; i < point_n - 1; i++)
		{
			for (int j = 1; j < point_n - 1; j++)
			{
				real x = i * delta_x;
				real y = j * delta_y;

				ez[i * point_n + j] = std::sin(pi * x) * std::sin(pi * y);
			}
		}
	} // #pragma omp parallel

	double compute_time = wall_time() - time_start;

	//Initial Ez (t = 0), not timed
	write_csv("ez_initial.csv", ez, point_n, delta_x, delta_y);

	time_start = wall_time();

	//FDTD
#pragma omp parallel
	{
		for (int t = 0; t < timestep_n; t++)
		{
#pragma omp for schedule(static) nowait
			for (int i = 0; i < point_n; i++)
			{
				for (int j = 1; j < point_n; j++)
				{
					hx[i * point_n + j] -= cy * (ez[i * point_n + j] - ez[i * point_n + j - 1]);
				}
			}

#pragma omp for schedule(static)
			for (int i = 1; i < point_n; i++)
			{
				for (int j = 0; j < point_n; j++)
				{
					hy[i * point_n + j] += cx * (ez[i * point_n + j] - ez[(i - 1) * point_n + j]);
				}
			}

#pragma omp for schedule(static)
			for (int i = 1; i < point_n - 1; i++)
			{
				for (int j = 1; j < point_n - 1; j++)
				{
					ez[i * point_n + j] += cx * (hy[(i + 1) * point_n + j] - hy[i * point_n + j]) 
											- cy * (hx[i * point_n + j + 1] - hx[i * point_n + j]);
				}
			}
		}
	} // #pragma omp parallel

	const real t_end = timestep_n * delta_t;
	const real cos_wt = std::cos(omega * t_end);

	const real ez_center = ez[n * point_n + n];
	const real exact_center = cos_wt;

	real sum_sq = 0;

#pragma omp parallel for schedule(static) reduction(+ : sum_sq)
	for (int i = 1; i < point_n - 1; i++)
	{
		for (int j = 1; j < point_n - 1; j++)
		{
			real x = i * delta_x;
			real y = j * delta_y;
			real exact = std::sin(pi * x) * std::sin(pi * y) * cos_wt;
			real diff = ez[i * point_n + j] - exact;

			sum_sq += diff * diff;
		}
	}

	const int interior_n = (point_n - 2) * (point_n - 2);
	const real rms_error = std::sqrt(sum_sq / interior_n);

	compute_time += wall_time() - time_start;

	std::printf("version      : openmp, %d threads\n", omp_get_max_threads());
	std::printf("grid         : %d x %d cells (n = %d)\n", 2 * n, 2 * n, n);
	std::printf("dx, dt       : %g, %g (Courant c*dt/dx = %g)\n", delta_x, delta_t, delta_t / delta_x);
	std::printf("time steps   : %d, t = %.6f\n", timestep_n, t_end);
	std::printf("Ez(0.5, 0.5) : %.6f (exact %.6f)\n", ez_center, exact_center);
	std::printf("RMS error    : %.4e (interior nodes)\n", rms_error);
	std::printf("compute time : %.7f s\n", compute_time);

	//Final Ez (t = timestep_n * delta_t), not timed
	write_csv("ez_final.csv", ez, point_n, delta_x, delta_y);

	std::free(hx);
	std::free(hy);
	std::free(ez);

	return 0;
}

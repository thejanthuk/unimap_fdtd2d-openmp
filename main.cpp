#include <cmath>
#include <cstdlib>

using real = double;

constexpr real pi = 3.14159265358979323846L;

int main(int argc, char **argv)
{
	const real length = 1;

	const int n = std::atoi(argv[1]);

	const int point_n = n * 2 + 1;
	const int timestep_n = std::atoi(argv[2]);

	const real stable_coeff = std::atof(argv[3]);

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
	}
} // #pragma omp parallel

	return 0;
}

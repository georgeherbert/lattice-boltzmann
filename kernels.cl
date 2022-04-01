#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9

typedef struct
{
    float speeds[NSPEEDS];
} t_speed;

__kernel void accelerate_flow(
    __global t_speed* cells,
    __global int* obstacles,
    int nx,
    int ny,
    float density,
    float accel) {

    /* compute weighting factors */
    float w1 = density * accel / 9.0;
    float w2 = density * accel / 36.0;

    /* modify the 2nd row of the grid */
    int jj = ny - 2;

    /* get column index */
    int ii = get_global_id(0);

    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj* nx] 
        && (cells[ii + jj * nx].speeds[3] - w1) > 0.f
        && (cells[ii + jj * nx].speeds[6] - w2) > 0.f
        && (cells[ii + jj * nx].speeds[7] - w2) > 0.f) {

        /* increase 'east-side' densities */
        cells[ii + jj * nx].speeds[1] += w1;
        cells[ii + jj * nx].speeds[5] += w2;
        cells[ii + jj * nx].speeds[8] += w2;
        /* decrease 'west-side' densities */
        cells[ii + jj * nx].speeds[3] -= w1;
        cells[ii + jj * nx].speeds[6] -= w2;
        cells[ii + jj * nx].speeds[7] -= w2;
    }
}

__kernel void timestep(
    __global t_speed* cells,
    __global t_speed* cells_new,
    __global int* obstacles,
    __local float* av_vels_local,
    __global float* av_vels_global,
    int nx,
    int ny,
    float omega) {

    const float c_sq_r = 3.f;
    const float two_c_sq_r = 1.5f;
    const float two_c_sq_sq_r = 4.5f;
    const float w0 = 4.f / 9.f; /* weighting factor */
    const float w1 = 1.f / 9.f; /* weighting factor */
    const float w2 = 1.f / 36.f; /* weighting factor */

    /* get column and row indices */
    int ii = get_global_id(0);
    int jj = get_global_id(1);

    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int y_n = (jj + 1) % ny;
    int x_e = (ii + 1) % nx;
    int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
    int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

    const float local_density = cells[ii + jj * nx].speeds[0] + cells[x_w + jj * nx].speeds[1] + cells[ii + y_s * nx].speeds[2] + cells[x_e + jj * nx].speeds[3] + cells[ii + y_n * nx].speeds[4] + cells[x_w + y_s * nx].speeds[5] + cells[x_e + y_s * nx].speeds[6] + cells[x_e + y_n * nx].speeds[7] + cells[x_w + y_n * nx].speeds[8];

    /* compute x and y velocity component */
    const float u_x = (cells[x_w + jj * nx].speeds[1] + cells[x_w + y_s * nx].speeds[5] + cells[x_w + y_n * nx].speeds[8] - (cells[x_e + jj * nx].speeds[3] + cells[x_e + y_s * nx].speeds[6] + cells[x_e + y_n * nx].speeds[7])) / local_density;
    const float u_y = (cells[ii + y_s * nx].speeds[2] + cells[x_w + y_s * nx].speeds[5] + cells[x_e + y_s * nx].speeds[6] - (cells[ii + y_n * nx].speeds[4] + cells[x_e + y_n * nx].speeds[7] + cells[x_w + y_n * nx].speeds[8])) / local_density;

    /* velocity squared */
    const float u_sq = u_x * u_x + u_y * u_y;

    /* relaxation step and obstacles step combined */
    cells_new[ii + jj * nx].speeds[0] = obstacles[ii + jj * nx] ? cells[ii + jj * nx].speeds[0] : cells[ii + jj * nx].speeds[0] + omega * (w0 * local_density * (1.f - u_sq * two_c_sq_r) - cells[ii + jj * nx].speeds[0]);
    cells_new[ii + jj * nx].speeds[1] = obstacles[ii + jj * nx] ? cells[x_e + jj * nx].speeds[3] : cells[x_w + jj * nx].speeds[1] + omega * (w1 * local_density * (1.f + u_x * c_sq_r + (u_x * u_x) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[x_w + jj * nx].speeds[1]);
    cells_new[ii + jj * nx].speeds[2] = obstacles[ii + jj * nx] ? cells[ii + y_n * nx].speeds[4] : cells[ii + y_s * nx].speeds[2] + omega * (w1 * local_density * (1.f + u_y * c_sq_r + (u_y * u_y) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[ii + y_s * nx].speeds[2]);
    cells_new[ii + jj * nx].speeds[3] = obstacles[ii + jj * nx] ? cells[x_w + jj * nx].speeds[1] : cells[x_e + jj * nx].speeds[3] + omega * (w1 * local_density * (1.f + -u_x * c_sq_r + (-u_x * -u_x) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[x_e + jj * nx].speeds[3]);
    cells_new[ii + jj * nx].speeds[4] = obstacles[ii + jj * nx] ? cells[ii + y_s * nx].speeds[2] : cells[ii + y_n * nx].speeds[4] + omega * (w1 * local_density * (1.f + -u_y * c_sq_r + (-u_y * -u_y) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[ii + y_n * nx].speeds[4]);
    cells_new[ii + jj * nx].speeds[5] = obstacles[ii + jj * nx] ? cells[x_e + y_n * nx].speeds[7] : cells[x_w + y_s * nx].speeds[5] + omega * (w2 * local_density * (1.f + (u_x + u_y) * c_sq_r + ((u_x + u_y) * (u_x + u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[x_w + y_s * nx].speeds[5]);
    cells_new[ii + jj * nx].speeds[6] = obstacles[ii + jj * nx] ? cells[x_w + y_n * nx].speeds[8] : cells[x_e + y_s * nx].speeds[6] + omega * (w2 * local_density * (1.f + (-u_x + u_y) * c_sq_r + ((-u_x + u_y) * (-u_x + u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[x_e + y_s * nx].speeds[6]);
    cells_new[ii + jj * nx].speeds[7] = obstacles[ii + jj * nx] ? cells[x_w + y_s * nx].speeds[5] : cells[x_e + y_n * nx].speeds[7] + omega * (w2 * local_density * (1.f + (-u_x - u_y) * c_sq_r + ((-u_x - u_y) * (-u_x - u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[x_e + y_n * nx].speeds[7]);
    cells_new[ii + jj * nx].speeds[8] = obstacles[ii + jj * nx] ? cells[x_e + y_s * nx].speeds[6] : cells[x_w + y_n * nx].speeds[8] + omega * (w2 * local_density * (1.f + (u_x - u_y) * c_sq_r + ((u_x - u_y) * (u_x - u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[x_w + y_n * nx].speeds[8]);

    int ny_local = get_local_size(0);
    int nx_local = get_local_size(1);
    int y_local_id = get_local_id(0);
    int x_local_id = get_local_id(1);

    int ny_global = get_global_size(0);
    int nx_global = get_global_size(1);

    int y_group_id = get_group_id(0);
    int x_group_id = get_group_id(1);

    av_vels_local[x_local_id + y_local_id * nx_local] = obstacles[ii + jj * nx] ? 0 : sqrt(u_sq);
    // printf("%f\n", sqrt(u_sq));

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x_local_id == 0 && y_local_id == 0) {
        float sum = 0.0f;

        for (int i = 0; i < ny_local * nx_local; i++) {
            sum += av_vels_local[i];
        }

        av_vels_global[x_group_id + y_group_id * (nx_global / nx_local)] = sum;
        // printf("%f %f\n", sum, av_vels_global[group_id]);
    }
}

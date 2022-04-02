#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9

typedef struct {
  float* restrict speeds_0;
  float* restrict speeds_1;
  float* restrict speeds_2;
  float* restrict speeds_3;
  float* restrict speeds_4;
  float* restrict speeds_5;
  float* restrict speeds_6;
  float* restrict speeds_7;
  float* restrict speeds_8;
} t_speed;

__kernel void accelerate_flow(
    __global float* restrict cells_speeds_0,
    __global float* restrict cells_speeds_1,
    __global float* restrict cells_speeds_2,
    __global float* restrict cells_speeds_3,
    __global float* restrict cells_speeds_4,
    __global float* restrict cells_speeds_5,
    __global float* restrict cells_speeds_6,
    __global float* restrict cells_speeds_7,
    __global float* restrict cells_speeds_8,
    __global int* restrict obstacles,
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
        && (cells_speeds_3[ii + jj * nx] - w1) > 0.f
        && (cells_speeds_6[ii + jj * nx] - w2) > 0.f
        && (cells_speeds_7[ii + jj * nx] - w2) > 0.f) {

        /* increase 'east-side' densities */
        cells_speeds_1[ii + jj * nx] += w1;
        cells_speeds_5[ii + jj * nx] += w2;
        cells_speeds_8[ii + jj * nx] += w2;
        /* decrease 'west-side' densities */
        cells_speeds_3[ii + jj * nx] -= w1;
        cells_speeds_6[ii + jj * nx] -= w2;
        cells_speeds_7[ii + jj * nx] -= w2;
    }
}

__kernel void timestep(
    __global float* restrict cells_speeds_0,
    __global float* restrict cells_speeds_1,
    __global float* restrict cells_speeds_2,
    __global float* restrict cells_speeds_3,
    __global float* restrict cells_speeds_4,
    __global float* restrict cells_speeds_5,
    __global float* restrict cells_speeds_6,
    __global float* restrict cells_speeds_7,
    __global float* restrict cells_speeds_8,
    __global int* restrict obstacles,
    __local float* restrict av_vels_local,
    __global float* restrict av_vels_global,
    int nx,
    int ny,
    float omega,
    int tt
    ) {

    const float c_sq_r = 3.f;
    const float two_c_sq_r = 1.5f;
    const float two_c_sq_sq_r = 4.5f;
    const float w0 = 4.f / 9.f; /* weighting factor */
    const float w1 = 1.f / 9.f; /* weighting factor */
    const float w2 = 1.f / 36.f; /* weighting factor */

    /* get column and row indices */
    const int ii = get_global_id(0);
    const int jj = get_global_id(1);

    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    const int y_n = (jj + 1) % ny;
    const int x_e = (ii + 1) % nx;
    const int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
    const int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

    const float local_density = cells_speeds_0[ii + jj * nx] + cells_speeds_1[x_w + jj * nx] + cells_speeds_2[ii + y_s * nx] + cells_speeds_3[x_e + jj * nx] + cells_speeds_4[ii + y_n * nx] + cells_speeds_5[x_w + y_s * nx] + cells_speeds_6[x_e + y_s * nx] + cells_speeds_7[x_e + y_n * nx] + cells_speeds_8[x_w + y_n * nx];

    /* compute x and y velocity component */
    const float u_x = (cells_speeds_1[x_w + jj * nx] + cells_speeds_5[x_w + y_s * nx] + cells_speeds_8[x_w + y_n * nx] - (cells_speeds_3[x_e + jj * nx] + cells_speeds_6[x_e + y_s * nx] + cells_speeds_7[x_e + y_n * nx])) / local_density;
    const float u_y = (cells_speeds_2[ii + y_s * nx] + cells_speeds_5[x_w + y_s * nx] + cells_speeds_6[x_e + y_s * nx] - (cells_speeds_4[ii + y_n * nx] + cells_speeds_7[x_e + y_n * nx] + cells_speeds_8[x_w + y_n * nx])) / local_density;

    /* velocity squared */
    const float u_sq = u_x * u_x + u_y * u_y;

    /* relaxation step and obstacles step combined */
    const float temp_speeds_0 = obstacles[ii + jj * nx] ? cells_speeds_0[ii + jj * nx] : cells_speeds_0[ii + jj * nx] + omega * (w0 * local_density * (1.f - u_sq * two_c_sq_r) - cells_speeds_0[ii + jj * nx]);
    const float temp_speeds_1 = obstacles[ii + jj * nx] ? cells_speeds_3[x_e + jj * nx] : cells_speeds_1[x_w + jj * nx] + omega * (w1 * local_density * (1.f + u_x * c_sq_r + (u_x * u_x) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells_speeds_1[x_w + jj * nx]);
    const float temp_speeds_2 = obstacles[ii + jj * nx] ? cells_speeds_4[ii + y_n * nx] : cells_speeds_2[ii + y_s * nx] + omega * (w1 * local_density * (1.f + u_y * c_sq_r + (u_y * u_y) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells_speeds_2[ii + y_s * nx]);
    const float temp_speeds_3 = obstacles[ii + jj * nx] ? cells_speeds_1[x_w + jj * nx] : cells_speeds_3[x_e + jj * nx] + omega * (w1 * local_density * (1.f + -u_x * c_sq_r + (-u_x * -u_x) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells_speeds_3[x_e + jj * nx]);
    const float temp_speeds_4 = obstacles[ii + jj * nx] ? cells_speeds_2[ii + y_s * nx] : cells_speeds_4[ii + y_n * nx] + omega * (w1 * local_density * (1.f + -u_y * c_sq_r + (-u_y * -u_y) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells_speeds_4[ii + y_n * nx]);
    const float temp_speeds_5 = obstacles[ii + jj * nx] ? cells_speeds_7[x_e + y_n * nx] : cells_speeds_5[x_w + y_s * nx] + omega * (w2 * local_density * (1.f + (u_x + u_y) * c_sq_r + ((u_x + u_y) * (u_x + u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells_speeds_5[x_w + y_s * nx]);
    const float temp_speeds_6 = obstacles[ii + jj * nx] ? cells_speeds_8[x_w + y_n * nx] : cells_speeds_6[x_e + y_s * nx] + omega * (w2 * local_density * (1.f + (-u_x + u_y) * c_sq_r + ((-u_x + u_y) * (-u_x + u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells_speeds_6[x_e + y_s * nx]);
    const float temp_speeds_7 = obstacles[ii + jj * nx] ? cells_speeds_5[x_w + y_s * nx] : cells_speeds_7[x_e + y_n * nx] + omega * (w2 * local_density * (1.f + (-u_x - u_y) * c_sq_r + ((-u_x - u_y) * (-u_x - u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells_speeds_7[x_e + y_n * nx]);
    const float temp_speeds_8 = obstacles[ii + jj * nx] ? cells_speeds_6[x_e + y_s * nx] : cells_speeds_8[x_w + y_n * nx] + omega * (w2 * local_density * (1.f + (u_x - u_y) * c_sq_r + ((u_x - u_y) * (u_x - u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells_speeds_8[x_w + y_n * nx]);

    cells_speeds_0[ii + jj * nx] = temp_speeds_0;
    cells_speeds_1[ii + jj * nx] = temp_speeds_1;
    cells_speeds_2[ii + jj * nx] = temp_speeds_2;
    cells_speeds_3[ii + jj * nx] = temp_speeds_3;
    cells_speeds_4[ii + jj * nx] = temp_speeds_4;
    cells_speeds_5[ii + jj * nx] = temp_speeds_5;
    cells_speeds_6[ii + jj * nx] = temp_speeds_6;
    cells_speeds_7[ii + jj * nx] = temp_speeds_7;
    cells_speeds_8[ii + jj * nx] = temp_speeds_8;

    const int ny_local = get_local_size(0);
    const int nx_local = get_local_size(1);
    const int y_local_id = get_local_id(0);
    const int x_local_id = get_local_id(1);

    const int ny_group = get_num_groups(0);
    const int nx_group = get_num_groups(1);
    const int y_group_id = get_group_id(0);
    const int x_group_id = get_group_id(1);

    int local_index = x_local_id + y_local_id * nx_local;
    int group_index = x_group_id + y_group_id * nx_group;

    av_vels_local[local_index] = obstacles[ii + jj * nx] ? 0 : sqrt(u_sq);

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_index == 0) {
        float sum = 0.0f;

        for (int i = 0; i < ny_local * nx_local; i++) {
            sum += av_vels_local[i];
        }

        av_vels_global[tt * (ny_group * nx_group) + group_index] = sum;
    }
}

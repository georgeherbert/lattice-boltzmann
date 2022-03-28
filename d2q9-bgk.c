/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>

#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE "av_vels.dat"
#define MASTER 0

/* struct to hold the parameter values */
typedef struct {
    int nx; /* no. of cells in x-direction */
    int ny; /* no. of cells in y-direction */
    int maxIters; /* no. of iterations */
    int reynolds_dim; /* dimension for Reynolds number */
    float density; /* density per link */
    float accel; /* density redistribution */
    float omega; /* relaxation parameter */
    float num_non_obstacles_r;
    int size;
    int rank;
    int rank_up;
    int rank_down;
    int rank_accelerate;
    int buffer_up_accelerate;
    int buffer_down_accelerate;
    int index_start;
    int index_stop;
    int num_rows;
    int *num_rows_per_rank;
    int *index_start_per_rank;
} t_param;

/* struct to hold the 'speed' values */
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

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile, t_param* params, t_speed** cells_ptr, t_speed** cells_new_ptr, 
    int** obstacles_ptr, int ** obstacles_output, float** av_vels_ptr, float** av_vels_buffer, float** send_row_buffer, 
    float** receive_row_buffer, float** send_section_buffer, float** receive_section_buffer, t_speed** cells_complete);

/* allocates rows to the different processors */
void allocate_rows(t_param* params);

/* the main calculation methods. */
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
float timestep(const t_param params, const t_speed* restrict cells, t_speed* restrict cells_new, const int* restrict obstacles);
void halo_exchange(const t_param* params, t_speed* cells, float* send_row_buffer, float* receive_row_buffer, const int tag, MPI_Status* status);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);
void collate(const t_param* params, t_speed* cells, t_speed* cells_complete, float* send_section_buffer, 
    float* receive_section_buffer, float* av_vels, float* av_vels_buffer, const int tag, MPI_Status* status);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** cells_new_ptr, int** obstacles_ptr, int ** obstacles_output,
    float** av_vels_ptr, float** av_vels_buffer, float** send_row_buffer, float** receive_row_buffer, float** send_section_buffer,
    float** receive_section_buffer, t_speed** cells_complete);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[]) {
    char* paramfile = NULL; /* name of the input parameter file */
    char* obstaclefile = NULL; /* name of a the input obstacle file */
    t_param params; /* struct to hold parameter values */
    t_speed* cells = NULL; /* grid containing fluid densities */
    t_speed* cells_new = NULL; /* scratch space */
    float* send_row_buffer = NULL;
    float* receive_row_buffer = NULL;
    float* send_section_buffer = NULL;
    float* receive_section_buffer = NULL;
    t_speed* cells_complete = NULL;
    int* obstacles = NULL; /* grid indicating which cells are blocked */
    int* obstacles_output = NULL;
    float* av_vels = NULL; /* a record of the av. velocity computed for each timestep */
    float* av_vels_buffer = NULL;
    struct timeval timstr; /* structure to hold elapsed time */
    double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */
    MPI_Status status;
    int tag = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &(params.size));
    MPI_Comm_rank(MPI_COMM_WORLD, &(params.rank));

    /* parse the command line */
    if (argc != 3) {
        usage(argv[0]);
    }
    else {
        paramfile = argv[1];
        obstaclefile = argv[2];
    }

    /* Total/init time starts here: initialise our data structures and load values from file */
    gettimeofday(&timstr, NULL);
    tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    init_tic=tot_tic;

    initialise(paramfile, obstaclefile, &params, &cells, &cells_new, &obstacles, &obstacles_output, &av_vels, &av_vels_buffer,
        &send_row_buffer, &receive_row_buffer, &send_section_buffer, &receive_section_buffer, &cells_complete);

    /* Init time stops here, compute time starts*/
    gettimeofday(&timstr, NULL);
    init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    comp_tic = init_toc;

    for (int tt = 0; tt < params.maxIters; tt++) {
        if (params.rank_accelerate || params.buffer_up_accelerate || params.buffer_down_accelerate) accelerate_flow(params, cells, obstacles);
        av_vels[tt] = timestep(params, cells, cells_new, obstacles);
        t_speed* temporary = cells;
        cells = cells_new;
        cells_new = temporary;
        halo_exchange(&params, cells, send_row_buffer, receive_row_buffer, tag, &status);
#ifdef DEBUG
        printf("==timestep: %d==\n", tt);
        printf("av velocity: %.12E\n", av_vels[tt]);
        printf("tot density: %.12E\n", total_density(params, cells));
#endif
    }
    
    /* Compute time stops here, collate time starts*/
    gettimeofday(&timstr, NULL);
    comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    col_tic=comp_toc;

    // Collate data from ranks here
    collate(&params, cells, cells_complete, send_section_buffer, receive_section_buffer, av_vels, av_vels_buffer, tag, &status);

    /* Total/collate time stops here.*/
    gettimeofday(&timstr, NULL);
    col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    tot_toc = col_toc;
    
    /* write final values and free memory */
    if (params.rank == MASTER) {
        printf("==done==\n");
        printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells_complete, obstacles_output));
        printf("Elapsed Init time:\t\t\t%.6lf (s)\n", init_toc - init_tic);
        printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
        printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc    - col_tic);
        printf("Elapsed Total time:\t\t\t%.6lf (s)\n", tot_toc    - tot_tic);
        write_values(params, cells_complete, obstacles_output, av_vels);
    }
    finalise(&params, &cells, &cells_new, &obstacles, &obstacles_output, &av_vels, &av_vels_buffer, &send_row_buffer, 
        &receive_row_buffer, &send_section_buffer, &receive_section_buffer, &cells_complete);
    MPI_Finalize();

    return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles) {
    /* compute weighting factors */
    float w1 = params.density * params.accel / 9.f;
    float w2 = params.density * params.accel / 36.f;

    /* modify the 2nd row of the grid */
    int jj;
    if (params.rank_accelerate && params.num_rows > 1) jj = params.num_rows - 1;
    if (params.rank_accelerate && params.num_rows == 1) jj = params.num_rows;
    if (params.buffer_up_accelerate) jj = params.num_rows + 1;
    if (params.buffer_down_accelerate) jj = 0;

    for (int ii = 0; ii < params.nx; ii++) {
        /* if the cell is not occupied and
        ** we don't send a negative density */
        if (!obstacles[ii + jj * params.nx]
            && (cells->speeds_3[ii + jj * params.nx] - w1) > 0.f
            && (cells->speeds_6[ii + jj * params.nx] - w2) > 0.f
            && (cells->speeds_7[ii + jj * params.nx] - w2) > 0.f) {
            /* increase 'east-side' densities */
            cells->speeds_1[ii + jj * params.nx] += w1;
            cells->speeds_5[ii + jj * params.nx] += w2;
            cells->speeds_8[ii + jj * params.nx] += w2;
            /* decrease 'west-side' densities */
            cells->speeds_3[ii + jj * params.nx] -= w1;
            cells->speeds_6[ii + jj * params.nx] -= w2;
            cells->speeds_7[ii + jj * params.nx] -= w2;
        }
    }

    return EXIT_SUCCESS;
}

float timestep(const t_param params, const t_speed* cells, t_speed* cells_new, const int* obstacles) {
    const float c_sq_r = 3.f;
    const float two_c_sq_r = 1.5f;
    const float two_c_sq_sq_r = 4.5f;
    const float w0 = 4.f / 9.f; /* weighting factor */
    const float w1 = 1.f / 9.f; /* weighting factor */
    const float w2 = 1.f / 36.f; /* weighting factor */
    float tot_u = 0.f; /* accumulated magnitudes of velocity for each cell */

    /* loop over the cells in the grid */
    // #pragma omp parallel for schedule(static), reduction(+:tot_u)
    for (int jj = 1; jj < params.num_rows + 1; jj++) {
        /* determine indices of north and south axis-direction neighbours 
        ** respecting periodic boundary conditions (wrap around) */
        const int y_n = jj + 1;
        const int y_s = jj - 1;
        #pragma omp simd reduction(+:tot_u)
        for (int ii = 0; ii < params.nx; ii++) {
            /* determine indices of east and west axis-direction neighbours 
            ** respecting periodic boundary conditions (wrap around) */
            const int x_e = (ii + 1) % params.nx;
            const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

            __assume(params.nx % 16 == 0);
            __assume_aligned(cells, 64);
            __assume_aligned(cells->speeds_0, 64);
            __assume_aligned(cells->speeds_1, 64);
            __assume_aligned(cells->speeds_2, 64);
            __assume_aligned(cells->speeds_3, 64);
            __assume_aligned(cells->speeds_4, 64);
            __assume_aligned(cells->speeds_5, 64);
            __assume_aligned(cells->speeds_6, 64);
            __assume_aligned(cells->speeds_7, 64);
            __assume_aligned(cells->speeds_8, 64);
            __assume_aligned(cells_new, 64);
            __assume_aligned(cells_new->speeds_0, 64);
            __assume_aligned(cells_new->speeds_1, 64);
            __assume_aligned(cells_new->speeds_2, 64);
            __assume_aligned(cells_new->speeds_3, 64);
            __assume_aligned(cells_new->speeds_4, 64);
            __assume_aligned(cells_new->speeds_5, 64);
            __assume_aligned(cells_new->speeds_6, 64);
            __assume_aligned(cells_new->speeds_7, 64);
            __assume_aligned(cells_new->speeds_8, 64);
            __assume_aligned(obstacles, 64);

            /* compute local density total */
            const float local_density = cells->speeds_0[ii + jj * params.nx] + cells->speeds_1[x_w + jj * params.nx] + cells->speeds_2[ii + y_s * params.nx] + cells->speeds_3[x_e + jj * params.nx] + cells->speeds_4[ii + y_n * params.nx] + cells->speeds_5[x_w + y_s * params.nx] + cells->speeds_6[x_e + y_s * params.nx] + cells->speeds_7[x_e + y_n * params.nx] + cells->speeds_8[x_w + y_n * params.nx];

            /* compute x and y velocity component */
            const float u_x = (cells->speeds_1[x_w + jj * params.nx] + cells->speeds_5[x_w + y_s * params.nx] + cells->speeds_8[x_w + y_n * params.nx] - (cells->speeds_3[x_e + jj * params.nx] + cells->speeds_6[x_e + y_s * params.nx] + cells->speeds_7[x_e + y_n * params.nx])) / local_density;
            const float u_y = (cells->speeds_2[ii + y_s * params.nx] + cells->speeds_5[x_w + y_s * params.nx] + cells->speeds_6[x_e + y_s * params.nx] - (cells->speeds_4[ii + y_n * params.nx] + cells->speeds_7[x_e + y_n * params.nx] + cells->speeds_8[x_w + y_n * params.nx])) / local_density;

            /* velocity squared */
            const float u_sq = u_x * u_x + u_y * u_y;

            /* relaxation step and obstacles step combined */
            cells_new->speeds_0[ii + jj * params.nx] = obstacles[ii + jj * params.nx] ? cells->speeds_0[ii + jj * params.nx] : cells->speeds_0[ii + jj * params.nx] + params.omega * (w0 * local_density * (1.f - u_sq * two_c_sq_r) - cells->speeds_0[ii + jj * params.nx]);
            cells_new->speeds_1[ii + jj * params.nx] = obstacles[ii + jj * params.nx] ? cells->speeds_3[x_e + jj * params.nx] : cells->speeds_1[x_w + jj * params.nx] + params.omega * (w1 * local_density * (1.f + u_x * c_sq_r + (u_x * u_x) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells->speeds_1[x_w + jj * params.nx]);
            cells_new->speeds_2[ii + jj * params.nx] = obstacles[ii + jj * params.nx] ? cells->speeds_4[ii + y_n * params.nx] : cells->speeds_2[ii + y_s * params.nx] + params.omega * (w1 * local_density * (1.f + u_y * c_sq_r + (u_y * u_y) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells->speeds_2[ii + y_s * params.nx]);
            cells_new->speeds_3[ii + jj * params.nx] = obstacles[ii + jj * params.nx] ? cells->speeds_1[x_w + jj * params.nx] : cells->speeds_3[x_e + jj * params.nx] + params.omega * (w1 * local_density * (1.f + -u_x * c_sq_r + (-u_x * -u_x) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells->speeds_3[x_e + jj * params.nx]);
            cells_new->speeds_4[ii + jj * params.nx] = obstacles[ii + jj * params.nx] ? cells->speeds_2[ii + y_s * params.nx] : cells->speeds_4[ii + y_n * params.nx] + params.omega * (w1 * local_density * (1.f + -u_y * c_sq_r + (-u_y * -u_y) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells->speeds_4[ii + y_n * params.nx]);
            cells_new->speeds_5[ii + jj * params.nx] = obstacles[ii + jj * params.nx] ? cells->speeds_7[x_e + y_n * params.nx] : cells->speeds_5[x_w + y_s * params.nx] + params.omega * (w2 * local_density * (1.f + (u_x + u_y) * c_sq_r + ((u_x + u_y) * (u_x + u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells->speeds_5[x_w + y_s * params.nx]);
            cells_new->speeds_6[ii + jj * params.nx] = obstacles[ii + jj * params.nx] ? cells->speeds_8[x_w + y_n * params.nx] : cells->speeds_6[x_e + y_s * params.nx] + params.omega * (w2 * local_density * (1.f + (-u_x + u_y) * c_sq_r + ((-u_x + u_y) * (-u_x + u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells->speeds_6[x_e + y_s * params.nx]);
            cells_new->speeds_7[ii + jj * params.nx] = obstacles[ii + jj * params.nx] ? cells->speeds_5[x_w + y_s * params.nx] : cells->speeds_7[x_e + y_n * params.nx] + params.omega * (w2 * local_density * (1.f + (-u_x - u_y) * c_sq_r + ((-u_x - u_y) * (-u_x - u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells->speeds_7[x_e + y_n * params.nx]);
            cells_new->speeds_8[ii + jj * params.nx] = obstacles[ii + jj * params.nx] ? cells->speeds_6[x_e + y_s * params.nx] : cells->speeds_8[x_w + y_n * params.nx] + params.omega * (w2 * local_density * (1.f + (u_x - u_y) * c_sq_r + ((u_x - u_y) * (u_x - u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells->speeds_8[x_w + y_n * params.nx]);
            
            tot_u += obstacles[ii + jj * params.nx] ? 0 : sqrtf(u_sq);
        }
    }
    return tot_u;
}

void halo_exchange(const t_param* params, t_speed* cells, float* send_row_buffer, float* receive_row_buffer, const int tag, MPI_Status* status) {
    // Send down receive up
    for (int ii = 0; ii < params->nx; ii++) {
        send_row_buffer[ii * 9] = cells->speeds_0[ii + params->nx];
        send_row_buffer[ii * 9 + 1] = cells->speeds_1[ii + params->nx];
        send_row_buffer[ii * 9 + 2] = cells->speeds_2[ii + params->nx];
        send_row_buffer[ii * 9 + 3] = cells->speeds_3[ii + params->nx];
        send_row_buffer[ii * 9 + 4] = cells->speeds_4[ii + params->nx];
        send_row_buffer[ii * 9 + 5] = cells->speeds_5[ii + params->nx];
        send_row_buffer[ii * 9 + 6] = cells->speeds_6[ii + params->nx];
        send_row_buffer[ii * 9 + 7] = cells->speeds_7[ii + params->nx];
        send_row_buffer[ii * 9 + 8] = cells->speeds_8[ii + params->nx];
    }

    MPI_Sendrecv(
        send_row_buffer, params->nx * 9, MPI_FLOAT, params->rank_down, tag, 
        receive_row_buffer, params->nx * 9, MPI_FLOAT, params->rank_up, tag,
        MPI_COMM_WORLD, status);

    for (int ii = 0; ii < params->nx; ii++) {
        cells->speeds_0[ii + params->nx * (params->num_rows + 1)] = receive_row_buffer[ii * 9];
        cells->speeds_1[ii + params->nx * (params->num_rows + 1)] = receive_row_buffer[ii * 9 + 1];
        cells->speeds_2[ii + params->nx * (params->num_rows + 1)] = receive_row_buffer[ii * 9 + 2];
        cells->speeds_3[ii + params->nx * (params->num_rows + 1)] = receive_row_buffer[ii * 9 + 3];
        cells->speeds_4[ii + params->nx * (params->num_rows + 1)] = receive_row_buffer[ii * 9 + 4];
        cells->speeds_5[ii + params->nx * (params->num_rows + 1)] = receive_row_buffer[ii * 9 + 5];
        cells->speeds_6[ii + params->nx * (params->num_rows + 1)] = receive_row_buffer[ii * 9 + 6];
        cells->speeds_7[ii + params->nx * (params->num_rows + 1)] = receive_row_buffer[ii * 9 + 7];
        cells->speeds_8[ii + params->nx * (params->num_rows + 1)] = receive_row_buffer[ii * 9 + 8];
    }

    // Send up receive down
    for (int ii = 0; ii < params->nx; ii++) {
        send_row_buffer[ii * 9] = cells->speeds_0[ii + params->nx * params->num_rows];
        send_row_buffer[ii * 9 + 1] = cells->speeds_1[ii + params->nx * params->num_rows];
        send_row_buffer[ii * 9 + 2] = cells->speeds_2[ii + params->nx * params->num_rows];
        send_row_buffer[ii * 9 + 3] = cells->speeds_3[ii + params->nx * params->num_rows];
        send_row_buffer[ii * 9 + 4] = cells->speeds_4[ii + params->nx * params->num_rows];
        send_row_buffer[ii * 9 + 5] = cells->speeds_5[ii + params->nx * params->num_rows];
        send_row_buffer[ii * 9 + 6] = cells->speeds_6[ii + params->nx * params->num_rows];
        send_row_buffer[ii * 9 + 7] = cells->speeds_7[ii + params->nx * params->num_rows];
        send_row_buffer[ii * 9 + 8] = cells->speeds_8[ii + params->nx * params->num_rows];
    }

    MPI_Sendrecv(
        send_row_buffer, params->nx * 9, MPI_FLOAT, params->rank_up, tag, 
        receive_row_buffer, params->nx * 9, MPI_FLOAT, params->rank_down, tag,
        MPI_COMM_WORLD, status);

    for (int ii = 0; ii < params->nx; ii++) {
        cells->speeds_0[ii] = receive_row_buffer[ii * 9];
        cells->speeds_1[ii] = receive_row_buffer[ii * 9 + 1];
        cells->speeds_2[ii] = receive_row_buffer[ii * 9 + 2];
        cells->speeds_3[ii] = receive_row_buffer[ii * 9 + 3];
        cells->speeds_4[ii] = receive_row_buffer[ii * 9 + 4];
        cells->speeds_5[ii] = receive_row_buffer[ii * 9 + 5];
        cells->speeds_6[ii] = receive_row_buffer[ii * 9 + 6];
        cells->speeds_7[ii] = receive_row_buffer[ii * 9 + 7];
        cells->speeds_8[ii] = receive_row_buffer[ii * 9 + 8];
    }
}

void collate(const t_param* params, t_speed* cells, t_speed* cells_complete, float* send_section_buffer, float* receive_section_buffer, float* av_vels, float* av_vels_buffer, const int tag, MPI_Status* status) {
    if (params->rank == MASTER) {
        cells->speeds_0[0] = 0.f;
        for (int jj = 0; jj < params->num_rows; jj++) {
            for (int ii = 0; ii < params->nx; ii++) {
                cells_complete->speeds_0[ii + jj * params->nx] = cells->speeds_0[ii + (jj + 1) * params->nx];
                cells_complete->speeds_1[ii + jj * params->nx] = cells->speeds_1[ii + (jj + 1) * params->nx];
                cells_complete->speeds_2[ii + jj * params->nx] = cells->speeds_2[ii + (jj + 1) * params->nx];
                cells_complete->speeds_3[ii + jj * params->nx] = cells->speeds_3[ii + (jj + 1) * params->nx];
                cells_complete->speeds_4[ii + jj * params->nx] = cells->speeds_4[ii + (jj + 1) * params->nx];
                cells_complete->speeds_5[ii + jj * params->nx] = cells->speeds_5[ii + (jj + 1) * params->nx];
                cells_complete->speeds_6[ii + jj * params->nx] = cells->speeds_6[ii + (jj + 1) * params->nx];
                cells_complete->speeds_7[ii + jj * params->nx] = cells->speeds_7[ii + (jj + 1) * params->nx];
                cells_complete->speeds_8[ii + jj * params->nx] = cells->speeds_8[ii + (jj + 1) * params->nx];
            }
        }
        for (int rr = 1; rr < params->size; rr++) {
            MPI_Recv(receive_section_buffer, params->nx * params->num_rows_per_rank[rr] * 9, MPI_FLOAT, rr, tag, MPI_COMM_WORLD, status);

            for (int jj = 0; jj < params->num_rows_per_rank[rr]; jj++) {
                for (int ii = 0; ii < params->nx; ii++) {
                    cells_complete->speeds_0[ii + (jj + params->index_start_per_rank[rr]) * params->nx] = receive_section_buffer[ii * 9 + jj * (params->nx * 9)];
                    cells_complete->speeds_1[ii + (jj + params->index_start_per_rank[rr]) * params->nx] = receive_section_buffer[ii * 9 + 1 + jj * (params->nx * 9)];
                    cells_complete->speeds_2[ii + (jj + params->index_start_per_rank[rr]) * params->nx] = receive_section_buffer[ii * 9 + 2 + jj * (params->nx * 9)];
                    cells_complete->speeds_3[ii + (jj + params->index_start_per_rank[rr]) * params->nx] = receive_section_buffer[ii * 9 + 3 + jj * (params->nx * 9)];
                    cells_complete->speeds_4[ii + (jj + params->index_start_per_rank[rr]) * params->nx] = receive_section_buffer[ii * 9 + 4 + jj * (params->nx * 9)];
                    cells_complete->speeds_5[ii + (jj + params->index_start_per_rank[rr]) * params->nx] = receive_section_buffer[ii * 9 + 5 + jj * (params->nx * 9)];
                    cells_complete->speeds_6[ii + (jj + params->index_start_per_rank[rr]) * params->nx] = receive_section_buffer[ii * 9 + 6 + jj * (params->nx * 9)];
                    cells_complete->speeds_7[ii + (jj + params->index_start_per_rank[rr]) * params->nx] = receive_section_buffer[ii * 9 + 7 + jj * (params->nx * 9)];
                    cells_complete->speeds_8[ii + (jj + params->index_start_per_rank[rr]) * params->nx] = receive_section_buffer[ii * 9 + 8 + jj * (params->nx * 9)];
                }
            }
        }
    }
    else {
        for (int jj = 0; jj < params->num_rows; jj++) {
            for (int ii = 0; ii < params->nx; ii++) {
                send_section_buffer[ii * 9 + jj * (params->nx * 9)] = cells->speeds_0[ii + (jj + 1) * params->nx];
                send_section_buffer[ii * 9 + 1 + jj * (params->nx * 9)] = cells->speeds_1[ii + (jj + 1) * params->nx];
                send_section_buffer[ii * 9 + 2 + jj * (params->nx * 9)] = cells->speeds_2[ii + (jj + 1) * params->nx];
                send_section_buffer[ii * 9 + 3 + jj * (params->nx * 9)] = cells->speeds_3[ii + (jj + 1) * params->nx];
                send_section_buffer[ii * 9 + 4 + jj * (params->nx * 9)] = cells->speeds_4[ii + (jj + 1) * params->nx];
                send_section_buffer[ii * 9 + 5 + jj * (params->nx * 9)] = cells->speeds_5[ii + (jj + 1) * params->nx];
                send_section_buffer[ii * 9 + 6 + jj * (params->nx * 9)] = cells->speeds_6[ii + (jj + 1) * params->nx];
                send_section_buffer[ii * 9 + 7 + jj * (params->nx * 9)] = cells->speeds_7[ii + (jj + 1) * params->nx];
                send_section_buffer[ii * 9 + 8 + jj * (params->nx * 9)] = cells->speeds_8[ii + (jj + 1) * params->nx];
            }
        }
        MPI_Send(send_section_buffer, params->nx * params->num_rows * 9, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
    }

    if (params->rank == MASTER) {
        for (int rr = 1; rr < params->size; rr++) {
            MPI_Recv(av_vels_buffer, params->maxIters, MPI_FLOAT, rr, tag, MPI_COMM_WORLD, status);
            for (int tt = 0; tt < params->maxIters; tt++) {
                av_vels[tt] += av_vels_buffer[tt];
            }
        }
        for (int tt = 0; tt < params->maxIters; tt++) {
            av_vels[tt] *= params->num_non_obstacles_r;
        }
    }
    else {
        MPI_Send(av_vels, params->maxIters, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
    }
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles) {
    int tot_cells = 0; /* no. of cells used in calculation */
    float tot_u = 0.f; /* accumulated magnitudes of velocity for each cell */

    /* loop over all non-blocked cells */
    for (int jj = 0; jj < params.ny; jj++) {
        for (int ii = 0; ii < params.nx; ii++) {
            if (!obstacles[ii + jj * params.nx]) {
                float local_density = 0.f;
                local_density += cells->speeds_0[ii + jj * params.nx];
                local_density += cells->speeds_1[ii + jj * params.nx];
                local_density += cells->speeds_2[ii + jj * params.nx];
                local_density += cells->speeds_3[ii + jj * params.nx];
                local_density += cells->speeds_4[ii + jj * params.nx];
                local_density += cells->speeds_5[ii + jj * params.nx];
                local_density += cells->speeds_6[ii + jj * params.nx];
                local_density += cells->speeds_7[ii + jj * params.nx];
                local_density += cells->speeds_8[ii + jj * params.nx];

                float u_x = (cells->speeds_1[ii + jj*params.nx] + cells->speeds_5[ii + jj*params.nx] + cells->speeds_8[ii + jj*params.nx]
                    - (cells->speeds_3[ii + jj*params.nx] + cells->speeds_6[ii + jj*params.nx] + cells->speeds_7[ii + jj*params.nx]))
                    / local_density;
                float u_y = (cells->speeds_2[ii + jj*params.nx] + cells->speeds_5[ii + jj*params.nx] + cells->speeds_6[ii + jj*params.nx]
                    - (cells->speeds_4[ii + jj*params.nx] + cells->speeds_7[ii + jj*params.nx] + cells->speeds_8[ii + jj*params.nx]))
                    / local_density;
                tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
                ++tot_cells;
            }
        }
    }

    return tot_u / (float)tot_cells;
}

void allocate_rows(t_param* params) {
    int minimum_rows = params->ny / params->size;
    int remainder = params->ny % params->size;

    int cumulative = 0;
    for (int rr = 0; rr < params->size; rr++) {
        params->num_rows_per_rank[rr] = minimum_rows + (rr < remainder);
        params->index_start_per_rank[rr] = cumulative;
        cumulative += params->num_rows_per_rank[rr];
    }
    params->num_rows = params->num_rows_per_rank[params->rank];

    params->index_start = params->index_start_per_rank[params->rank];
    params->index_stop = params->index_start + params->num_rows;

    params->rank_accelerate = (params->ny - 2 >= params->index_start && params->ny - 2 < params->index_stop);
    params->buffer_up_accelerate = params->ny - 2 == params->index_stop;
    params->buffer_down_accelerate = params->ny - 2 == params->index_start - 1;
    params->rank_up = (params->rank + 1) % params->size;
    params->rank_down = ((params->rank - 1) % params->size + params->size) % params->size;
}

int initialise(const char* paramfile, const char* obstaclefile, t_param* params, t_speed** cells_ptr, t_speed** cells_new_ptr, 
    int** obstacles_ptr, int ** obstacles_output, float** av_vels_ptr, float** av_vels_buffer, float** send_row_buffer, 
    float** receive_row_buffer, float** send_section_buffer, float** receive_section_buffer, t_speed** cells_complete) {
    char message[1024]; /* message buffer */
    FILE* fp; /* file pointer */
    int xx, yy; /* generic array indices */
    int blocked; /* indicates whether a cell is blocked by an obstacle */
    int retval; /* to hold return value for checking */

    /* open the parameter file */
    fp = fopen(paramfile, "r");
    if (fp == NULL) {
        sprintf(message, "could not open input parameter file: %s", paramfile); 
        die(message, __LINE__, __FILE__);
    }

    /* read in the parameter values */
    retval = fscanf(fp, "%d\n", &(params->nx)); if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);
    retval = fscanf(fp, "%d\n", &(params->ny)); if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);
    retval = fscanf(fp, "%d\n", &(params->maxIters)); if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);
    retval = fscanf(fp, "%d\n", &(params->reynolds_dim)); if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);
    retval = fscanf(fp, "%f\n", &(params->density)); if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);
    retval = fscanf(fp, "%f\n", &(params->accel)); if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);
    retval = fscanf(fp, "%f\n", &(params->omega)); if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);
    fclose(fp);

    // Calculates the allocations for each rank

    params->num_rows_per_rank = malloc(sizeof(int) * params->size);
    params->index_start_per_rank = malloc(sizeof(int) * params->size);
    allocate_rows(params);
    // printf("\nSize: %d", params->size);
    // printf("\nRank: %d", params->rank);
    // printf("\nRank up: %d", params->rank_up);
    // printf("\nRank down: %d", params->rank_down);
    // printf("\nIndex start: %d", params->index_start);
    // printf("\nIndex stop: %d", params->index_stop);
    // printf("\nNumber of rows: %d\n\n", params->num_rows);

    /* main grid */
    *cells_ptr = (t_speed*)_mm_malloc(sizeof(float*) * NSPEEDS, 64);
    (*cells_ptr)->speeds_0 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_ptr)->speeds_1 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_ptr)->speeds_2 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_ptr)->speeds_3 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_ptr)->speeds_4 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_ptr)->speeds_5 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_ptr)->speeds_6 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_ptr)->speeds_7 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_ptr)->speeds_8 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);

    *cells_new_ptr = (t_speed*)_mm_malloc(sizeof(float*) * NSPEEDS, 64);
    (*cells_new_ptr)->speeds_0 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_new_ptr)->speeds_1 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_new_ptr)->speeds_2 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_new_ptr)->speeds_3 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_new_ptr)->speeds_4 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_new_ptr)->speeds_5 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_new_ptr)->speeds_6 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_new_ptr)->speeds_7 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);
    (*cells_new_ptr)->speeds_8 = (float*)_mm_malloc(sizeof(float) * ((params->num_rows + 2) * params->nx), 64);

    *cells_complete = (t_speed*)_mm_malloc(sizeof(float*) * NSPEEDS, 64);
    (*cells_complete)->speeds_0 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_complete)->speeds_1 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_complete)->speeds_2 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_complete)->speeds_3 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_complete)->speeds_4 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_complete)->speeds_5 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_complete)->speeds_6 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_complete)->speeds_7 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_complete)->speeds_8 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

    *obstacles_ptr = _mm_malloc(sizeof(int) * ((params->num_rows + 2) * params->nx), 64);
    *obstacles_output = _mm_malloc(sizeof(int) * (params->ny * params->nx), 64);

    /* initialise densities */
    float w0 = params->density * 4.f / 9.f;
    float w1 = params->density / 9.f;
    float w2 = params->density / 36.f;

    // #pragma omp parallel for schedule(static)
    for (int jj = 0; jj < params->num_rows + 2; jj++) {
        for (int ii = 0; ii < params->nx; ii++) {
            /* centre */
            (*cells_ptr)->speeds_0[ii + jj*params->nx] = w0;
            /* axis directions */
            (*cells_ptr)->speeds_1[ii + jj*params->nx] = w1;
            (*cells_ptr)->speeds_2[ii + jj*params->nx] = w1;
            (*cells_ptr)->speeds_3[ii + jj*params->nx] = w1;
            (*cells_ptr)->speeds_4[ii + jj*params->nx] = w1;
            /* diagonals */
            (*cells_ptr)->speeds_5[ii + jj*params->nx] = w2;
            (*cells_ptr)->speeds_6[ii + jj*params->nx] = w2;
            (*cells_ptr)->speeds_7[ii + jj*params->nx] = w2;
            (*cells_ptr)->speeds_8[ii + jj*params->nx] = w2;
        }
    }

    /* first set all cells in obstacle array to zero */
    // #pragma omp parallel for schedule(static)
    for (int jj = 0; jj < params->num_rows + 2; jj++) {
        for (int ii = 0; ii < params->nx; ii++) {
            (*obstacles_ptr)[ii + jj * params->nx] = 0;
        }
    }
    for (int jj = 0; jj < params->ny; jj++) {
        for (int ii = 0; ii < params->nx; ii++) {
            (*obstacles_output)[ii + jj * params->nx] = 0;
        }
    }

    /* open the obstacle data file */
    fp = fopen(obstaclefile, "r");
    if (fp == NULL) {
        sprintf(message, "could not open input obstacles file: %s", obstaclefile);
        die(message, __LINE__, __FILE__);
    }

    int num_obstacles = 0;
    /* read-in the blocked cells list */
    while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
        /* some checks */
        if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
        if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
        if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
        if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

        /* assign to array */
        if (yy >= params->index_start - 1 && yy < params->index_stop + 1) {
            (*obstacles_ptr)[xx + (yy - params->index_start + 1) * params->nx] = blocked;
        }
        (*obstacles_output)[xx + yy * params->nx] = blocked;
        ++num_obstacles;
    }
    params->num_non_obstacles_r = 1.f / (params->nx * params->ny - num_obstacles);

    fclose(fp);

    *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);
    *av_vels_buffer = (float*)malloc(sizeof(float) * params->maxIters);
    *send_row_buffer = malloc(sizeof(float) * params->nx * 9);
    *receive_row_buffer = malloc(sizeof(float) * params->nx * 9);
    *send_section_buffer = malloc(sizeof(float) * params->nx * params->num_rows * 9);
    *receive_section_buffer = malloc(sizeof(float) * params->nx * params->num_rows * 9);

    return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** cells_new_ptr,
    int** obstacles_ptr, int ** obstacles_output, float** av_vels_ptr, float** av_vels_buffer, float** send_row_buffer, float** receive_row_buffer,
    float** send_section_buffer, float** receive_section_buffer, t_speed** cells_complete) {
    _mm_free(*cells_ptr); *cells_ptr = NULL;
    _mm_free(*cells_new_ptr); *cells_new_ptr = NULL;
    _mm_free(*cells_complete); *cells_complete = NULL;
    _mm_free(*obstacles_ptr); *obstacles_ptr = NULL;
    _mm_free(*obstacles_output); *obstacles_output = NULL;
    free(*av_vels_ptr); *av_vels_ptr = NULL;
    free(*av_vels_buffer); *av_vels_buffer = NULL;
    free(*send_row_buffer); *send_row_buffer = NULL;
    free(*receive_row_buffer); *receive_row_buffer = NULL;
    free(*send_section_buffer); *send_section_buffer = NULL;
    free(*receive_section_buffer); *receive_section_buffer = NULL;
    free(params->num_rows_per_rank);
    free(params->index_start_per_rank);
    return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, t_speed* cells, int* obstacles) {
    const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
    return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells) {
    float total = 0.f;    /* accumulator */
    for (int jj = 0; jj < params.ny; jj++) {
        for (int ii = 0; ii < params.nx; ii++) {
            total += cells->speeds_0[ii + jj * params.nx];
            total += cells->speeds_1[ii + jj * params.nx];
            total += cells->speeds_2[ii + jj * params.nx];
            total += cells->speeds_3[ii + jj * params.nx];
            total += cells->speeds_4[ii + jj * params.nx];
            total += cells->speeds_5[ii + jj * params.nx];
            total += cells->speeds_6[ii + jj * params.nx];
            total += cells->speeds_7[ii + jj * params.nx];
            total += cells->speeds_8[ii + jj * params.nx];
        }
    }
    return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels) {
    FILE* fp; /* file pointer */
    const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
    float local_density; /* per grid cell sum of densities */
    float pressure; /* fluid pressure in grid cell */
    float u_x; /* x-component of velocity in grid cell */
    float u_y; /* y-component of velocity in grid cell */
    float u; /* norm--root of summed squares--of u_x and u_y */

    fp = fopen(FINALSTATEFILE, "w");
    if (fp == NULL) die("could not open file output file", __LINE__, __FILE__);

    for (int jj = 0; jj < params.ny; jj++) {
        for (int ii = 0; ii < params.nx; ii++) {
            /* an occupied cell */
            if (obstacles[ii + jj*params.nx]) {
                u_x = u_y = u = 0.f;
                pressure = params.density * c_sq;
            }
            /* no obstacle */
            else {
                local_density = 0.f;
                local_density += cells->speeds_0[ii + jj*params.nx];
                local_density += cells->speeds_1[ii + jj*params.nx];
                local_density += cells->speeds_2[ii + jj*params.nx];
                local_density += cells->speeds_3[ii + jj*params.nx];
                local_density += cells->speeds_4[ii + jj*params.nx];
                local_density += cells->speeds_5[ii + jj*params.nx];
                local_density += cells->speeds_6[ii + jj*params.nx];
                local_density += cells->speeds_7[ii + jj*params.nx];
                local_density += cells->speeds_8[ii + jj*params.nx];

                u_x = (cells->speeds_1[ii + jj*params.nx] + cells->speeds_5[ii + jj*params.nx] + cells->speeds_8[ii + jj*params.nx]
                    - (cells->speeds_3[ii + jj*params.nx] + cells->speeds_6[ii + jj*params.nx] + cells->speeds_7[ii + jj*params.nx]))
                    / local_density;
                u_y = (cells->speeds_2[ii + jj*params.nx] + cells->speeds_5[ii + jj*params.nx] + cells->speeds_6[ii + jj*params.nx]
                    - (cells->speeds_4[ii + jj*params.nx] + cells->speeds_7[ii + jj*params.nx] + cells->speeds_8[ii + jj*params.nx]))
                    / local_density;
                u = sqrtf((u_x * u_x) + (u_y * u_y));
                pressure = local_density * c_sq;
            }
            fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
        }
    }
    fclose(fp);

    fp = fopen(AVVELSFILE, "w");
    if (fp == NULL) die("could not open file output file", __LINE__, __FILE__);
    for (int ii = 0; ii < params.maxIters; ii++) {
        fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
    }
    fclose(fp);

    return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file) {
    fprintf(stderr, "Error at line %d of file %s:\n", line, file);
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    exit(EXIT_FAILURE);
}

void usage(const char* exe) {
    fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
    exit(EXIT_FAILURE);
}
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
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE "av_vels.dat"
#define OCLFILE "kernels.cl"
#define NX_LOCAL 32
#define NY_LOCAL 16

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
    int num_work_groups;
} t_param;

/* struct to hold OpenCL objects */
typedef struct {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    cl_program program;
    cl_kernel accelerate_flow;
    cl_kernel timestep;

    cl_mem cells_speeds_0;
    cl_mem cells_speeds_1;
    cl_mem cells_speeds_2;
    cl_mem cells_speeds_3;
    cl_mem cells_speeds_4;
    cl_mem cells_speeds_5;
    cl_mem cells_speeds_6;
    cl_mem cells_speeds_7;
    cl_mem cells_speeds_8;
    cl_mem obstacles;
    cl_mem av_vels_global;

} t_ocl;

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
int initialise(const char* paramfile, const char* obstaclefile, t_param* params, t_speed** cells_ptr, 
    int** obstacles_ptr, float** av_vels_ptr, float** av_vels_temp_ptr, t_ocl* ocl);

/*
** The main calculation methods.
*/
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, int** obstacles_ptr, float** av_vels_ptr, float** av_vels_temp_ptr, t_ocl ocl);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl);

/* utility functions */
void checkError(cl_int err, const char *op, const int line);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

cl_device_id selectOpenCLDevice();

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[]) {
    char* paramfile = NULL; /* name of the input parameter file */
    char* obstaclefile = NULL; /* name of a the input obstacle file */
    t_param params; /* struct to hold parameter values */
    t_ocl ocl; /* struct to hold OpenCL objects */
    t_speed* cells = NULL; /* grid containing fluid densities */
    int* obstacles = NULL; /* grid indicating which cells are blocked */
    float* av_vels = NULL; /* a record of the av. velocity computed for each timestep */
    float* av_vels_temp = NULL;
    cl_int err;
    struct timeval timstr; /* structure to hold elapsed time */
    double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

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
    init_tic = tot_tic;
    initialise(paramfile, obstaclefile, &params, &cells, &obstacles, &av_vels, &av_vels_temp, &ocl);

    /* Init time stops here, compute time starts*/
    gettimeofday(&timstr, NULL);
    init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    comp_tic = init_toc;

    // Write cells to OpenCL buffer
    err = clEnqueueWriteBuffer(ocl.queue, ocl.cells_speeds_0, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_0, 0, NULL, NULL); checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(ocl.queue, ocl.cells_speeds_1, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_1, 0, NULL, NULL); checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(ocl.queue, ocl.cells_speeds_2, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_2, 0, NULL, NULL); checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(ocl.queue, ocl.cells_speeds_3, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_3, 0, NULL, NULL); checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(ocl.queue, ocl.cells_speeds_4, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_4, 0, NULL, NULL); checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(ocl.queue, ocl.cells_speeds_5, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_5, 0, NULL, NULL); checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(ocl.queue, ocl.cells_speeds_6, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_6, 0, NULL, NULL); checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(ocl.queue, ocl.cells_speeds_7, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_7, 0, NULL, NULL); checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(ocl.queue, ocl.cells_speeds_8, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_8, 0, NULL, NULL); checkError(err, "writing cells data", __LINE__);

    // Write obstacles to OpenCL buffer
    err = clEnqueueWriteBuffer( ocl.queue, ocl.obstacles, CL_TRUE, 0, sizeof(cl_int) * params.nx * params.ny, obstacles, 0, NULL, NULL);
    checkError(err, "writing obstacles data", __LINE__);

    const int second_row = (params.ny - 2) * params.nx;
    const float density_mul_accel = params.density * params.accel;

    err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.cells_speeds_0); checkError(err, "setting accelerate_flow arg 0", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 1, sizeof(cl_mem), &ocl.cells_speeds_1); checkError(err, "setting accelerate_flow arg 1", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 2, sizeof(cl_mem), &ocl.cells_speeds_2); checkError(err, "setting accelerate_flow arg 2", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 3, sizeof(cl_mem), &ocl.cells_speeds_3); checkError(err, "setting accelerate_flow arg 3", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 4, sizeof(cl_mem), &ocl.cells_speeds_4); checkError(err, "setting accelerate_flow arg 4", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 5, sizeof(cl_mem), &ocl.cells_speeds_5); checkError(err, "setting accelerate_flow arg 5", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 6, sizeof(cl_mem), &ocl.cells_speeds_6); checkError(err, "setting accelerate_flow arg 6", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 7, sizeof(cl_mem), &ocl.cells_speeds_7); checkError(err, "setting accelerate_flow arg 7", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 8, sizeof(cl_mem), &ocl.cells_speeds_8); checkError(err, "setting accelerate_flow arg 8", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 9, sizeof(cl_mem), &ocl.obstacles); checkError(err, "setting accelerate_flow arg 9", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 10, sizeof(cl_int), &second_row); checkError(err, "setting accelerate_flow arg 10", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 11, sizeof(cl_float), &density_mul_accel); checkError(err, "setting accelerate_flow arg 11", __LINE__);

    err = clSetKernelArg(ocl.timestep, 0, sizeof(cl_mem), &ocl.cells_speeds_0); checkError(err, "setting timestep arg 0", __LINE__);
    err = clSetKernelArg(ocl.timestep, 1, sizeof(cl_mem), &ocl.cells_speeds_1); checkError(err, "setting timestep arg 1", __LINE__);
    err = clSetKernelArg(ocl.timestep, 2, sizeof(cl_mem), &ocl.cells_speeds_2); checkError(err, "setting timestep arg 2", __LINE__);
    err = clSetKernelArg(ocl.timestep, 3, sizeof(cl_mem), &ocl.cells_speeds_3); checkError(err, "setting timestep arg 3", __LINE__);
    err = clSetKernelArg(ocl.timestep, 4, sizeof(cl_mem), &ocl.cells_speeds_4); checkError(err, "setting timestep arg 4", __LINE__);
    err = clSetKernelArg(ocl.timestep, 5, sizeof(cl_mem), &ocl.cells_speeds_5); checkError(err, "setting timestep arg 5", __LINE__);
    err = clSetKernelArg(ocl.timestep, 6, sizeof(cl_mem), &ocl.cells_speeds_6); checkError(err, "setting timestep arg 6", __LINE__);
    err = clSetKernelArg(ocl.timestep, 7, sizeof(cl_mem), &ocl.cells_speeds_7); checkError(err, "setting timestep arg 7", __LINE__);
    err = clSetKernelArg(ocl.timestep, 8, sizeof(cl_mem), &ocl.cells_speeds_8); checkError(err, "setting timestep arg 8", __LINE__);
    err = clSetKernelArg(ocl.timestep, 9, sizeof(cl_mem), &ocl.obstacles); checkError(err, "setting timestep arg 9", __LINE__);
    err = clSetKernelArg(ocl.timestep, 10, sizeof(cl_float) * NX_LOCAL * NY_LOCAL, NULL); checkError(err, "setting timestep arg 10", __LINE__);
    err = clSetKernelArg(ocl.timestep, 11, sizeof(cl_mem), &ocl.av_vels_global); checkError(err, "setting timestep arg 11", __LINE__);
    err = clSetKernelArg(ocl.timestep, 12, sizeof(cl_int), &params.nx); checkError(err, "setting timestep arg 12", __LINE__);
    err = clSetKernelArg(ocl.timestep, 13, sizeof(cl_int), &params.ny); checkError(err, "setting timestep arg 13", __LINE__);
    err = clSetKernelArg(ocl.timestep, 14, sizeof(cl_float), &params.omega); checkError(err, "setting timestep arg 14", __LINE__);

    size_t global_accelerate_flow[1] = {params.nx};
    size_t global_timestep[2] = {params.nx, params.ny};
    size_t local_timestep[2] = {NX_LOCAL, NY_LOCAL};
    for (int tt = 0; tt < params.maxIters; tt++) {
        err = clEnqueueNDRangeKernel(ocl.queue, ocl.accelerate_flow, 1, NULL, global_accelerate_flow, NULL, 0, NULL, NULL); checkError(err, "enqueueing accelerate_flow kernel", __LINE__);
        err = clSetKernelArg(ocl.timestep, 15, sizeof(cl_int), &tt); checkError(err, "setting timestep arg 15", __LINE__);
        err = clEnqueueNDRangeKernel(ocl.queue, ocl.timestep, 2, NULL, global_timestep, local_timestep, 0, NULL, NULL); checkError(err, "enqueueing timestep kernel", __LINE__);
#ifdef DEBUG
        printf("==timestep: %d==\n", tt);
        printf("av velocity: %.12E\n", av_vels[tt]);
        printf("tot density: %.12E\n", total_density(params, cells));
#endif
    }

    err = clEnqueueReadBuffer(ocl.queue, ocl.cells_speeds_0, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_0, 0, NULL, NULL); checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(ocl.queue, ocl.cells_speeds_1, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_1, 0, NULL, NULL); checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(ocl.queue, ocl.cells_speeds_2, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_2, 0, NULL, NULL); checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(ocl.queue, ocl.cells_speeds_3, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_3, 0, NULL, NULL); checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(ocl.queue, ocl.cells_speeds_4, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_4, 0, NULL, NULL); checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(ocl.queue, ocl.cells_speeds_5, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_5, 0, NULL, NULL); checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(ocl.queue, ocl.cells_speeds_6, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_6, 0, NULL, NULL); checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(ocl.queue, ocl.cells_speeds_7, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_7, 0, NULL, NULL); checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(ocl.queue, ocl.cells_speeds_8, CL_TRUE, 0, sizeof(float) * params.nx * params.ny, cells->speeds_8, 0, NULL, NULL); checkError(err, "reading cells data", __LINE__);

    err = clEnqueueReadBuffer(ocl.queue, ocl.av_vels_global, CL_TRUE, 0, sizeof(float) * params.num_work_groups * params.maxIters, av_vels_temp, 0, NULL, NULL); checkError(err, "reading cells data", __LINE__);
    #pragma omp parallel for
    for (int i = 0; i < params.maxIters; i++) {
        av_vels[i] = 0.0f;
        for (int j = 0; j < params.num_work_groups; j++) {
            av_vels[i] += av_vels_temp[i * params.num_work_groups + j];
        }
        av_vels[i] *= params.num_non_obstacles_r;
    }

    /* Compute time stops here, collate time starts*/
    gettimeofday(&timstr, NULL);
    comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    col_tic=comp_toc;

    // Collate data from ranks here 

    /* Total/collate time stops here.*/
    gettimeofday(&timstr, NULL);
    col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    tot_toc = col_toc;

    /* write final values and free memory */
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles, ocl));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
    write_values(params, cells, obstacles, av_vels);
    finalise(&params, &cells, &obstacles, &av_vels, &av_vels_temp, ocl);

    return EXIT_SUCCESS;
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

int initialise(const char* paramfile, const char* obstaclefile, t_param* params, t_speed** cells_ptr,
    int** obstacles_ptr, float** av_vels_ptr, float** av_vels_temp_ptr, t_ocl *ocl) {
    char message[1024]; /* message buffer */
    FILE* fp; /* file pointer */
    int xx, yy; /* generic array indices */
    int blocked; /* indicates whether a cell is blocked by an obstacle */
    int retval; /* to hold return value for checking */
    char* ocl_src; /* OpenCL kernel source */
    long ocl_size; /* size of OpenCL kernel source */

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

    params->num_work_groups = (params->nx * params->ny) / (NX_LOCAL * NY_LOCAL);

    /* main grid */
    *cells_ptr = (t_speed*)_mm_malloc(sizeof(float*) * NSPEEDS, 64);
    (*cells_ptr)->speeds_0 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_ptr)->speeds_1 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_ptr)->speeds_2 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_ptr)->speeds_3 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_ptr)->speeds_4 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_ptr)->speeds_5 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_ptr)->speeds_6 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_ptr)->speeds_7 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    (*cells_ptr)->speeds_8 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

    *obstacles_ptr = _mm_malloc(sizeof(int) * params->ny * params->nx, 64);
    if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

    /* initialise densities */
    float w0 = params->density * 4.f / 9.f;
    float w1 = params->density / 9.f;
    float w2 = params->density / 36.f;

    for (int jj = 0; jj < params->ny; jj++) {
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
    for (int jj = 0; jj < params->ny; jj++) {
        for (int ii = 0; ii < params->nx; ii++) {
            (*obstacles_ptr)[ii + jj*params->nx] = 0;
        }
    }

    int num_obstacles = 0;
    /* open the obstacle data file */
    fp = fopen(obstaclefile, "r");
    if (fp == NULL) {
        sprintf(message, "could not open input obstacles file: %s", obstaclefile);
        die(message, __LINE__, __FILE__);
    }

    /* read-in the blocked cells list */
    while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
        /* some checks */
        if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
        if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
        if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
        if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

        /* assign to array */
        (*obstacles_ptr)[xx + yy*params->nx] = blocked;
        ++num_obstacles;
    }
    params->num_non_obstacles_r = 1.f / (params->nx * params->ny - num_obstacles);

    fclose(fp);

    *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);
    *av_vels_temp_ptr = (float*)malloc(sizeof(float) * params->maxIters * params->num_work_groups);

    // OpenCL setup
    cl_int err;
    ocl->device = selectOpenCLDevice();

    // Create OpenCL context
    ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
    checkError(err, "creating context", __LINE__);

    fp = fopen(OCLFILE, "r");
    if (fp == NULL) {
        sprintf(message, "could not open OpenCL kernel file: %s", OCLFILE);
        die(message, __LINE__, __FILE__);
    }

    // Create OpenCL command queue
    ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
    checkError(err, "creating command queue", __LINE__);

    // Load OpenCL kernel source
    fseek(fp, 0, SEEK_END);
    ocl_size = ftell(fp) + 1;
    ocl_src = (char*)malloc(ocl_size);
    memset(ocl_src, 0, ocl_size);
    fseek(fp, 0, SEEK_SET);
    fread(ocl_src, 1, ocl_size, fp);
    fclose(fp);

    // Create OpenCL program
    ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**)&ocl_src, NULL, &err);
    free(ocl_src);
    checkError(err, "creating program", __LINE__);

    // Build OpenCL program
    err = clBuildProgram(ocl->program, 1, &ocl->device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t sz;
        clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
        char *buildlog = malloc(sz);
        clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, sz, buildlog, NULL);
        fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
        free(buildlog);
    }
    checkError(err, "building program", __LINE__);

    // Create OpenCL kernels
    ocl->accelerate_flow = clCreateKernel(ocl->program, "accelerate_flow", &err);
    checkError(err, "creating accelerate_flow kernel", __LINE__);
    ocl->timestep = clCreateKernel(ocl->program, "timestep", &err);
    checkError(err, "creating timestep kernel", __LINE__);

    // Allocate OpenCL buffers
    ocl->cells_speeds_0 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * params->nx * params->ny, NULL, &err); checkError(err, "creating cells buffer", __LINE__);
    ocl->cells_speeds_1 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * params->nx * params->ny, NULL, &err); checkError(err, "creating cells buffer", __LINE__);
    ocl->cells_speeds_2 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * params->nx * params->ny, NULL, &err); checkError(err, "creating cells buffer", __LINE__);
    ocl->cells_speeds_3 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * params->nx * params->ny, NULL, &err); checkError(err, "creating cells buffer", __LINE__);
    ocl->cells_speeds_4 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * params->nx * params->ny, NULL, &err); checkError(err, "creating cells buffer", __LINE__);
    ocl->cells_speeds_5 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * params->nx * params->ny, NULL, &err); checkError(err, "creating cells buffer", __LINE__);
    ocl->cells_speeds_6 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * params->nx * params->ny, NULL, &err); checkError(err, "creating cells buffer", __LINE__);
    ocl->cells_speeds_7 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * params->nx * params->ny, NULL, &err); checkError(err, "creating cells buffer", __LINE__);
    ocl->cells_speeds_8 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * params->nx * params->ny, NULL, &err); checkError(err, "creating cells buffer", __LINE__);
    ocl->obstacles = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(cl_int) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating obstacles buffer", __LINE__);
    ocl->av_vels_global = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_float) * params->num_work_groups * params->maxIters, NULL, &err);
    checkError(err, "creating av_vels_global buffer", __LINE__);

    return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, int** obstacles_ptr, float** av_vels_ptr, float** av_vels_temp_ptr, t_ocl ocl) {
    _mm_free(*cells_ptr); *cells_ptr = NULL;
    _mm_free(*obstacles_ptr); *obstacles_ptr = NULL;
    free(*av_vels_ptr); *av_vels_ptr = NULL;
    free(*av_vels_temp_ptr); *av_vels_temp_ptr = NULL;

    clReleaseMemObject(ocl.cells_speeds_0);
    clReleaseMemObject(ocl.cells_speeds_1);
    clReleaseMemObject(ocl.cells_speeds_2);
    clReleaseMemObject(ocl.cells_speeds_3);
    clReleaseMemObject(ocl.cells_speeds_4);
    clReleaseMemObject(ocl.cells_speeds_5);
    clReleaseMemObject(ocl.cells_speeds_6);
    clReleaseMemObject(ocl.cells_speeds_7);
    clReleaseMemObject(ocl.cells_speeds_8);
    clReleaseMemObject(ocl.obstacles);
    clReleaseMemObject(ocl.av_vels_global);
    clReleaseKernel(ocl.accelerate_flow);
    clReleaseKernel(ocl.timestep);
    clReleaseProgram(ocl.program);
    clReleaseCommandQueue(ocl.queue);
    clReleaseContext(ocl.context);

    return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl) {
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
            fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
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

void checkError(cl_int err, const char *op, const int line) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
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

#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

cl_device_id selectOpenCLDevice() {
    cl_int err;
    cl_uint num_platforms = 0;
    cl_uint total_devices = 0;
    cl_platform_id platforms[8];
    cl_device_id devices[MAX_DEVICES];
    char name[MAX_DEVICE_NAME];

    // Get list of platforms
    err = clGetPlatformIDs(8, platforms, &num_platforms);
    checkError(err, "getting platforms", __LINE__);

    // Get list of devices
    for (cl_uint p = 0; p < num_platforms; p++) {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
            MAX_DEVICES-total_devices, devices+total_devices, &num_devices);
        checkError(err, "getting device name", __LINE__);
        total_devices += num_devices;
    }

    // Print list of devices
    printf("\nAvailable OpenCL devices:\n");
    for (cl_uint d = 0; d < total_devices; d++) {
        clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
        printf("%2d: %s\n", d, name);
    }
    printf("\n");

    // Use first device unless OCL_DEVICE environment variable used
    cl_uint device_index = 0;
    char *dev_env = getenv("OCL_DEVICE");
    if (dev_env) {
        char *end;
        device_index = strtol(dev_env, &end, 10);
        if (strlen(end)) die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
    }

    if (device_index >= total_devices) {
        fprintf(stderr, "device index set to %d but only %d devices available\n", device_index, total_devices);
        exit(1);
    }

    // Print OpenCL device name
    clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

    return devices[device_index];
}

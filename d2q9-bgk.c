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
  int index_start;
  int index_stop;
  int num_rows;
  int num_rows_extended;
} t_param;

/* struct to hold the 'speed' values */
typedef struct {
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile, t_param* params, 
  t_speed** cells_ptr, t_speed** cells_new_ptr, int** obstacles_ptr, float** av_vels_ptr, float** av_vels_buffer,
  t_speed** send_row_buffer, t_speed** receive_row_buffer, 
  t_speed** send_section_buffer, t_speed** receive_section_buffer, t_speed** cells_complete);

/* allocates rows to the different processors */
void allocate_rows(t_param* params);

/* the main calculation methods. */
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
float timestep(const t_param params, const t_speed* restrict cells, t_speed* restrict cells_new, const int* restrict obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** cells_new_ptr,
  int** obstacles_ptr, float** av_vels_ptr, float** av_vels_buffer, t_speed** send_row_buffer, t_speed** receive_row_buffer,
  t_speed** send_section_buffer, t_speed** receive_section_buffer, t_speed** cells_complete);

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
  t_speed* send_row_buffer = NULL;
  t_speed* receive_row_buffer = NULL;
  t_speed* send_section_buffer = NULL;
  t_speed* receive_section_buffer = NULL;
  t_speed* cells_complete = NULL;
  int* obstacles = NULL; /* grid indicating which cells are blocked */
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
  initialise(paramfile, obstaclefile, &params, &cells, &cells_new, &obstacles, &av_vels, &av_vels_buffer,
    &send_row_buffer, &receive_row_buffer, &send_section_buffer, &receive_section_buffer, &cells_complete);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic = init_toc;

  for (int tt = 0; tt < params.maxIters; tt++) {
    accelerate_flow(params, cells, obstacles);
    av_vels[tt] = timestep(params, cells, cells_new, obstacles);
    t_speed* temporary = cells;
    cells = cells_new;
    cells_new = temporary;

    // Send down receive up
    for (int ii = 0; ii < params.nx; ii++) {
      send_row_buffer[ii] = cells[ii + params.nx];
    }
    MPI_Sendrecv(
      send_row_buffer, params.nx * 9, MPI_FLOAT, params.rank_down, tag, 
      receive_row_buffer, params.nx * 9, MPI_FLOAT, params.rank_up, tag,
      MPI_COMM_WORLD, &status);
    for (int ii = 0; ii < params.nx; ii++) {
      cells[ii + params.nx * 65] = receive_row_buffer[ii];
    }

    // Send up receive down
    for (int ii = 0; ii < params.nx; ii++) {
      send_row_buffer[ii] = cells[ii + params.nx * 64];
    }
    MPI_Sendrecv(
      send_row_buffer, params.nx * 9, MPI_FLOAT, params.rank_up, tag, 
      receive_row_buffer, params.nx * 9, MPI_FLOAT, params.rank_down, tag,
      MPI_COMM_WORLD, &status);
    for (int ii = 0; ii < params.nx; ii++) {
      cells[ii] = receive_row_buffer[ii];
    }

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
  if (params.rank == 0) {
    MPI_Recv(receive_section_buffer, params.nx * 64 * 9, MPI_FLOAT, params.rank_down, tag, MPI_COMM_WORLD, &status);
    for (int jj = 0; jj < 64; jj++) {
      for (int ii = 0; ii < params.nx; ii++) {
        cells_complete[ii + (jj + 64) * params.nx] = cells[ii + (jj + 1) * params.nx];
      }
    }
    for (int jj = 0; jj < 64; jj++) {
      for (int ii = 0; ii < params.nx; ii++) {
        cells_complete[ii + jj * params.nx] = receive_section_buffer[ii + jj * params.nx];
      }
    }
  }
  else if (params.rank == 1) {
    for (int jj = 0; jj < 64; jj++) {
      for (int ii = 0; ii < params.nx; ii++) {
        send_section_buffer[ii + jj * params.nx] = cells[ii + (jj + 1) * params.nx];
      }
    }
    MPI_Send(send_section_buffer, params.nx * 64 * 9, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
  }

  if (params.rank == MASTER) {
    MPI_Recv(av_vels_buffer, params.maxIters, MPI_FLOAT, params.rank_down, tag, MPI_COMM_WORLD, &status);
    for (int tt = 0; tt < params.maxIters; tt++) {
      av_vels[tt] += av_vels_buffer[tt];
    }
  }
  else if (params.rank == 1) {
    MPI_Send(av_vels, params.maxIters, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
  }

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  if (params.rank == 0) {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells_complete, obstacles));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n", init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n", tot_toc  - tot_tic);
    write_values(params, cells_complete, obstacles, av_vels);
  }
  finalise(&params, &cells, &cells_new, &obstacles, &av_vels, &av_vels_buffer, &send_row_buffer, 
    &receive_row_buffer, &send_section_buffer, &receive_section_buffer, &cells_complete);
  MPI_Finalize();

  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles) {
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  if (params.rank == 0) {
    int jj = 63;

    for (int ii = 0; ii < params.nx; ii++) {
      /* if the cell is not occupied and
      ** we don't send a negative density */
      if (!obstacles[ii + 126 * params.nx]
        && (cells[ii + jj*params.nx].speeds[3] - w1) > 0.f
        && (cells[ii + jj*params.nx].speeds[6] - w2) > 0.f
        && (cells[ii + jj*params.nx].speeds[7] - w2) > 0.f) {
        /* increase 'east-side' densities */
        cells[ii + jj*params.nx].speeds[1] += w1;
        cells[ii + jj*params.nx].speeds[5] += w2;
        cells[ii + jj*params.nx].speeds[8] += w2;
        /* decrease 'west-side' densities */
        cells[ii + jj*params.nx].speeds[3] -= w1;
        cells[ii + jj*params.nx].speeds[6] -= w2;
        cells[ii + jj*params.nx].speeds[7] -= w2;
      }
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
  for (int jj = 1; jj < 65; jj++) {
    /* determine indices of north and south axis-direction neighbours 
    ** respecting periodic boundary conditions (wrap around) */
    const int y_n = jj + 1;
    const int y_s = jj - 1;
    for (int ii = 0; ii < params.nx; ii++) {
      /* determine indices of east and west axis-direction neighbours 
      ** respecting periodic boundary conditions (wrap around) */
      const int x_e = (ii + 1) % params.nx;
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

      /* if the cell contains an obstacle */
      int obstacles_jj;
      if (params.rank == 0) {
        obstacles_jj = jj + 63;
      }
      else if (params.rank == 1) {
        obstacles_jj = jj;
      }

      if (obstacles[ii + obstacles_jj*params.nx]) {
        /* run after propagate stage, so taking values from speed variables
        ** mirroring, and writing into cells_new grid */
        cells_new[ii + jj*params.nx].speeds[0] = cells[ii + jj * params.nx].speeds[0];
        cells_new[ii + jj*params.nx].speeds[1] = cells[x_e + jj * params.nx].speeds[3];
        cells_new[ii + jj*params.nx].speeds[2] = cells[ii + y_n * params.nx].speeds[4];
        cells_new[ii + jj*params.nx].speeds[3] = cells[x_w + jj * params.nx].speeds[1];
        cells_new[ii + jj*params.nx].speeds[4] = cells[ii + y_s * params.nx].speeds[2];
        cells_new[ii + jj*params.nx].speeds[5] = cells[x_e + y_n * params.nx].speeds[7];
        cells_new[ii + jj*params.nx].speeds[6] = cells[x_w + y_n * params.nx].speeds[8];
        cells_new[ii + jj*params.nx].speeds[7] = cells[x_w + y_s * params.nx].speeds[5];
        cells_new[ii + jj*params.nx].speeds[8] = cells[x_e + y_s * params.nx].speeds[6];
      }
      /* don't consider occupied cells */
      else {
        /* compute local density total */
        const float local_density = cells[ii + jj * params.nx].speeds[0] + cells[x_w + jj * params.nx].speeds[1] + cells[ii + y_s * params.nx].speeds[2] + cells[x_e + jj * params.nx].speeds[3] + cells[ii + y_n * params.nx].speeds[4] + cells[x_w + y_s * params.nx].speeds[5] + cells[x_e + y_s * params.nx].speeds[6] + cells[x_e + y_n * params.nx].speeds[7] + cells[x_w + y_n * params.nx].speeds[8];

        /* compute x and y velocity component */
        const float u_x = (cells[x_w + jj * params.nx].speeds[1] + cells[x_w + y_s * params.nx].speeds[5] + cells[x_w + y_n * params.nx].speeds[8] - (cells[x_e + jj * params.nx].speeds[3] + cells[x_e + y_s * params.nx].speeds[6] + cells[x_e + y_n * params.nx].speeds[7])) / local_density;
        const float u_y = (cells[ii + y_s * params.nx].speeds[2] + cells[x_w + y_s * params.nx].speeds[5] + cells[x_e + y_s * params.nx].speeds[6] - (cells[ii + y_n * params.nx].speeds[4] + cells[x_e + y_n * params.nx].speeds[7] + cells[x_w + y_n * params.nx].speeds[8])) / local_density;

        /* velocity squared */
        const float u_sq = u_x * u_x + u_y * u_y;

        /* relaxation step */
        cells_new[ii + jj * params.nx].speeds[0] = cells[ii + jj * params.nx].speeds[0] + params.omega * (w0 * local_density * (1.f - u_sq * two_c_sq_r) - cells[ii + jj * params.nx].speeds[0]);
        cells_new[ii + jj * params.nx].speeds[1] = cells[x_w + jj * params.nx].speeds[1] + params.omega * (w1 * local_density * (1.f + u_x * c_sq_r + (u_x * u_x) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[x_w + jj * params.nx].speeds[1]);
        cells_new[ii + jj * params.nx].speeds[2] = cells[ii + y_s * params.nx].speeds[2] + params.omega * (w1 * local_density * (1.f + u_y * c_sq_r + (u_y * u_y) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[ii + y_s * params.nx].speeds[2]);
        cells_new[ii + jj * params.nx].speeds[3] = cells[x_e + jj * params.nx].speeds[3] + params.omega * (w1 * local_density * (1.f + -u_x * c_sq_r + (-u_x * -u_x) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[x_e + jj * params.nx].speeds[3]);
        cells_new[ii + jj * params.nx].speeds[4] = cells[ii + y_n * params.nx].speeds[4] + params.omega * (w1 * local_density * (1.f + -u_y * c_sq_r + (-u_y * -u_y) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[ii + y_n * params.nx].speeds[4]);
        cells_new[ii + jj * params.nx].speeds[5] = cells[x_w + y_s * params.nx].speeds[5] + params.omega * (w2 * local_density * (1.f + (u_x + u_y) * c_sq_r + ((u_x + u_y) * (u_x + u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[x_w + y_s * params.nx].speeds[5]);
        cells_new[ii + jj * params.nx].speeds[6] = cells[x_e + y_s * params.nx].speeds[6] + params.omega * (w2 * local_density * (1.f + (-u_x + u_y) * c_sq_r + ((-u_x + u_y) * (-u_x + u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[x_e + y_s * params.nx].speeds[6]);
        cells_new[ii + jj * params.nx].speeds[7] = cells[x_e + y_n * params.nx].speeds[7] + params.omega * (w2 * local_density * (1.f + (-u_x - u_y) * c_sq_r + ((-u_x - u_y) * (-u_x - u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[x_e + y_n * params.nx].speeds[7]);
        cells_new[ii + jj * params.nx].speeds[8] = cells[x_w + y_n * params.nx].speeds[8] + params.omega * (w2 * local_density * (1.f + (u_x - u_y) * c_sq_r + ((u_x - u_y) * (u_x - u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells[x_w + y_n * params.nx].speeds[8]);
        
        tot_u += sqrtf(u_sq);
      }
    }
  }
  return tot_u * params.num_non_obstacles_r;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles) {
  int tot_cells = 0; /* no. of cells used in calculation */
  float tot_u = 0.f; /* accumulated magnitudes of velocity for each cell */

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx]) {
        /* local density total */
        float local_density = 0.f;
        local_density += cells[ii + jj*params.nx].speeds[0];
        local_density += cells[ii + jj*params.nx].speeds[1];
        local_density += cells[ii + jj*params.nx].speeds[2];
        local_density += cells[ii + jj*params.nx].speeds[3];
        local_density += cells[ii + jj*params.nx].speeds[4];
        local_density += cells[ii + jj*params.nx].speeds[5];
        local_density += cells[ii + jj*params.nx].speeds[6];
        local_density += cells[ii + jj*params.nx].speeds[7];
        local_density += cells[ii + jj*params.nx].speeds[8];

        /* x-component of velocity */
        float u_x = (cells[ii + jj*params.nx].speeds[1] + cells[ii + jj*params.nx].speeds[5] + cells[ii + jj*params.nx].speeds[8]
          - (cells[ii + jj*params.nx].speeds[3] + cells[ii + jj*params.nx].speeds[6] + cells[ii + jj*params.nx].speeds[7]))
          / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj*params.nx].speeds[2] + cells[ii + jj*params.nx].speeds[5] + cells[ii + jj*params.nx].speeds[6]
          - (cells[ii + jj*params.nx].speeds[4] + cells[ii + jj*params.nx].speeds[7] + cells[ii + jj*params.nx].speeds[8]))
          / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

void allocate_rows(t_param* params) {
  int minimum_rows = params->ny / params->size;
  int remainder = params->ny % params->size;
  if (params->rank < remainder) {
    params->index_start = (minimum_rows + 1) * params->rank;
    params->index_stop = params->index_start + minimum_rows + 1;
  }
  else if (params->rank == remainder) { 
    params->index_start = (minimum_rows + 1) * params->rank;
    params->index_stop = params->index_start + minimum_rows;
  }
  else {
    params->index_start = remainder + minimum_rows * params->rank;
    params->index_stop = params->index_start + minimum_rows;
  }
  params->num_rows = params->index_stop - params->index_start;
  params->num_rows_extended = params->num_rows + 2;

  params->rank_up = ((params->rank - 1) % params->size + params->size) % params->size;
  params->rank_down = (params->rank + 1) % params->size;
}

int initialise(const char* paramfile, const char* obstaclefile, t_param* params, 
  t_speed** cells_ptr, t_speed** cells_new_ptr, int** obstacles_ptr, float** av_vels_ptr, float** av_vels_buffer,
  t_speed** send_row_buffer, t_speed** receive_row_buffer,
  t_speed** send_section_buffer, t_speed** receive_section_buffer, t_speed** cells_complete) {
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
  retval = fscanf(fp, "%d\n", &(params->nx));
  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));
  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));
  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));
  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));
  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));
  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));
  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  // Calculates the allocations for each rank
  allocate_rows(params);
  printf("\nSize: %d", params->size);
  printf("\nRank: %d", params->rank);
  printf("\nRank up: %d", params->rank_up);
  printf("\nRank down: %d", params->rank_down);
  printf("\nIndex start: %d", params->index_start);
  printf("\nIndex stop: %d", params->index_stop);
  printf("\nNumber of rows: %d", params->num_rows);
  printf("\nNumber of extended rows: %d\n", params->num_rows_extended);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (66 * params->nx));
  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *cells_new_ptr = (t_speed*)malloc(sizeof(t_speed) * (66 * params->nx));
  if (*cells_new_ptr == NULL) die("cannot allocate memory for cells_new", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));
  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density / 9.f;
  float w2 = params->density / 36.f;

  // #pragma omp parallel for schedule(static)
  for (int jj = 0; jj < 66; jj++) {
    for (int ii = 0; ii < params->nx; ii++) {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  // #pragma omp parallel for schedule(static)
  for (int jj = 0; jj < params->ny; jj++) {
    for (int ii = 0; ii < params->nx; ii++) {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
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
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
    ++num_obstacles;
  }
  params->num_non_obstacles_r = 1.f / (params->nx * params->ny - num_obstacles);

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);
  *av_vels_buffer = (float*)malloc(sizeof(float) * params->maxIters);

  *send_row_buffer = malloc(sizeof(t_speed) * params->nx);
  *receive_row_buffer  = malloc(sizeof(t_speed) * params->nx);

  *send_section_buffer = malloc(sizeof(t_speed) * params->nx * 64);
  *receive_section_buffer = malloc(sizeof(t_speed) * params->nx * 64);
  *cells_complete = malloc(sizeof(t_speed) * params->nx * params->ny);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** cells_new_ptr,
  int** obstacles_ptr, float** av_vels_ptr, float** av_vels_buffer, t_speed** send_row_buffer, t_speed** receive_row_buffer,
  t_speed** send_section_buffer, t_speed** receive_section_buffer, t_speed** cells_complete) {
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*cells_new_ptr);
  *cells_new_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  free(*av_vels_buffer);
  *av_vels_buffer = NULL;

  free(*send_row_buffer);
  *send_row_buffer = NULL;

  free(*receive_row_buffer);
  *receive_row_buffer = NULL;

  free(*send_section_buffer);
  *send_section_buffer = NULL;

  free(*receive_section_buffer);
  *receive_section_buffer = NULL;

  free(*cells_complete);
  *cells_complete = NULL;

  return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, t_speed* cells, int* obstacles) {
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells) {
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      total += cells[ii + jj*params.nx].speeds[0];
      total += cells[ii + jj*params.nx].speeds[1];
      total += cells[ii + jj*params.nx].speeds[2];
      total += cells[ii + jj*params.nx].speeds[3];
      total += cells[ii + jj*params.nx].speeds[4];
      total += cells[ii + jj*params.nx].speeds[5];
      total += cells[ii + jj*params.nx].speeds[6];
      total += cells[ii + jj*params.nx].speeds[7];
      total += cells[ii + jj*params.nx].speeds[8];
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

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

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
        local_density += cells[ii + jj*params.nx].speeds[0];
        local_density += cells[ii + jj*params.nx].speeds[1];
        local_density += cells[ii + jj*params.nx].speeds[2];
        local_density += cells[ii + jj*params.nx].speeds[3];
        local_density += cells[ii + jj*params.nx].speeds[4];
        local_density += cells[ii + jj*params.nx].speeds[5];
        local_density += cells[ii + jj*params.nx].speeds[6];
        local_density += cells[ii + jj*params.nx].speeds[7];
        local_density += cells[ii + jj*params.nx].speeds[8];

        /* compute x velocity component */
        u_x = (cells[ii + jj*params.nx].speeds[1] + cells[ii + jj*params.nx].speeds[5] + cells[ii + jj*params.nx].speeds[8]
          - (cells[ii + jj*params.nx].speeds[3] + cells[ii + jj*params.nx].speeds[6] + cells[ii + jj*params.nx].speeds[7]))
          / local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2] + cells[ii + jj*params.nx].speeds[5] + cells[ii + jj*params.nx].speeds[6]
          - (cells[ii + jj*params.nx].speeds[4] + cells[ii + jj*params.nx].speeds[7] + cells[ii + jj*params.nx].speeds[8]))
          / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

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
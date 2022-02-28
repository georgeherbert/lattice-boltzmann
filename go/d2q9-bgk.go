package main

import (
	"fmt"
	"os"
	"bufio"
	"strconv"
	"strings"
	"time"
	"math"
)

type t_param struct {
	nx int
	ny int
	maxIters int
	reynolds_dim int
	density float32
	accel float32
	omega float32
	num_non_obstacles_r float32
  }

type t_speed struct {
	speeds_0[] float32
	speeds_1[] float32
	speeds_2[] float32
	speeds_3[] float32
	speeds_4[] float32
	speeds_5[] float32
	speeds_6[] float32
	speeds_7[] float32
	speeds_8[] float32
};

func initialise(paramfile string, obstaclefile string) (t_speed, t_speed, []int, t_param) {
	var params t_param;

	file_params, _ := os.Open(paramfile)
	scanner_params := bufio.NewScanner(file_params); scanner_params.Scan()
	params.nx, _ = strconv.Atoi(scanner_params.Text()); scanner_params.Scan()
	params.ny, _ = strconv.Atoi(scanner_params.Text()); scanner_params.Scan()
	params.maxIters, _ = strconv.Atoi(scanner_params.Text()); scanner_params.Scan()
	params.reynolds_dim, _ = strconv.Atoi(scanner_params.Text()); scanner_params.Scan()
	params_density_float64, _ := strconv.ParseFloat(scanner_params.Text(), 32); scanner_params.Scan()
	params_accel_float_64, _ := strconv.ParseFloat(scanner_params.Text(), 32); scanner_params.Scan()
	params_omega_float_64, _ := strconv.ParseFloat(scanner_params.Text(), 32)
	params.density = float32(params_density_float64)
	params.accel = float32(params_accel_float_64)
	params.omega = float32(params_omega_float_64)
	file_params.Close()

	w0 := float32(params.density * 4 / 9)
  	w1 := float32(params.density / 9)
  	w2 := float32(params.density / 36)

	var cells t_speed;
	var cells_new t_speed;

	cells.speeds_0 = make([]float32, params.nx * params.ny)
	cells.speeds_1 = make([]float32, params.nx * params.ny)
	cells.speeds_2 = make([]float32, params.nx * params.ny)
	cells.speeds_3 = make([]float32, params.nx * params.ny)
	cells.speeds_4 = make([]float32, params.nx * params.ny)
	cells.speeds_5 = make([]float32, params.nx * params.ny)
	cells.speeds_6 = make([]float32, params.nx * params.ny)
	cells.speeds_7 = make([]float32, params.nx * params.ny)
	cells.speeds_8 = make([]float32, params.nx * params.ny)

	cells_new.speeds_0 = make([]float32, params.nx * params.ny)
	cells_new.speeds_1 = make([]float32, params.nx * params.ny)
	cells_new.speeds_2 = make([]float32, params.nx * params.ny)
	cells_new.speeds_3 = make([]float32, params.nx * params.ny)
	cells_new.speeds_4 = make([]float32, params.nx * params.ny)
	cells_new.speeds_5 = make([]float32, params.nx * params.ny)
	cells_new.speeds_6 = make([]float32, params.nx * params.ny)
	cells_new.speeds_7 = make([]float32, params.nx * params.ny)
	cells_new.speeds_8 = make([]float32, params.nx * params.ny)

	for jj := 0; jj < params.ny; jj++ {
		for ii := 0; ii < params.nx; ii++ {
			/* centre */
			cells.speeds_0[ii + jj * params.nx] = w0;
			/* axis directions */
			cells.speeds_1[ii + jj * params.nx] = w1;
			cells.speeds_2[ii + jj * params.nx] = w1;
			cells.speeds_3[ii + jj * params.nx] = w1;
			cells.speeds_4[ii + jj * params.nx] = w1;
			/* diagonals */
			cells.speeds_5[ii + jj * params.nx] = w2;
			cells.speeds_6[ii + jj * params.nx] = w2;
			cells.speeds_7[ii + jj * params.nx] = w2;
			cells.speeds_8[ii + jj * params.nx] = w2;
		}
	}

	obstacles := make([]int, params.nx * params.ny)

	for jj := 0; jj < params.ny; jj++ {
		for ii := 0; ii < params.nx; ii++ {
			obstacles[ii + jj * params.nx] = 0;
		}
	}

	num_obstacles := 0
	file_obstacles, _ := os.Open(obstaclefile)
	scanner_obstacles := bufio.NewScanner(file_obstacles)
	for scanner_obstacles.Scan() {
		line_split := strings.Fields(scanner_obstacles.Text())
		xx, _ := strconv.Atoi(line_split[0])
		yy, _ := strconv.Atoi(line_split[1])
		blocked, _ := strconv.Atoi(line_split[2])
		obstacles[xx + yy * params.nx] = blocked
		num_obstacles++
	}
	params.num_non_obstacles_r = float32(1.0 / float32(params.nx * params.ny - num_obstacles))

	return cells, cells_new, obstacles, params
}

func accelerate_flow(params t_param, cells *t_speed, obstacles []int) {
	w1 := params.density * params.accel / 9
	w2 := params.density * params.accel / 36
	
	jj := params.ny - 2

	for ii := 0; ii < params.nx; ii++ {
		if obstacles[ii + jj*params.nx] == 0 && (cells.speeds_3[ii + jj*params.nx] - w1) > 0 && (cells.speeds_6[ii + jj*params.nx] - w2) > 0 && (cells.speeds_7[ii + jj*params.nx] - w2) > 0 {
			cells.speeds_1[ii + jj*params.nx] += w1
			cells.speeds_5[ii + jj*params.nx] += w2
			cells.speeds_8[ii + jj*params.nx] += w2
			cells.speeds_3[ii + jj*params.nx] -= w1
			cells.speeds_6[ii + jj*params.nx] -= w2
			cells.speeds_7[ii + jj*params.nx] -= w2
		}
	}
}

func timestep(params t_param, cells *t_speed, cells_new *t_speed, obstacles []int) float32 {
	c_sq_r := float32(3.0)
	two_c_sq_r := float32(1.5)
	two_c_sq_sq_r := float32(4.5)
	w0 := float32(4.0 / 9.0)
	w1 := float32(1.0 / 9.0)
	w2 := float32(1.0 / 36.0)
	tot_u := float32(0.0)
  
	for jj := 0; jj < params.ny; jj++ {
		y_n := (jj + 1) % params.ny
		var y_s int
		if jj == 0 {
			y_s = jj + params.ny - 1
		} else {
			y_s = jj - 1
		}
		for ii := 0; ii < params.nx; ii++ {
			x_e := (ii + 1) % params.nx;
			var x_w int
			if ii == 0 {
				x_w = ii + params.nx - 1
			} else {
				x_w = ii - 1
			}
	
			if obstacles[ii + jj * params.nx] == 1 {
				cells_new.speeds_0[ii + jj * params.nx] = cells.speeds_0[ii + jj * params.nx]
				cells_new.speeds_1[ii + jj * params.nx] = cells.speeds_3[x_e + jj * params.nx] 
				cells_new.speeds_2[ii + jj * params.nx] = cells.speeds_4[ii + y_n * params.nx] 
				cells_new.speeds_3[ii + jj * params.nx] = cells.speeds_1[x_w + jj * params.nx] 
				cells_new.speeds_4[ii + jj * params.nx] = cells.speeds_2[ii + y_s * params.nx] 
				cells_new.speeds_5[ii + jj * params.nx] = cells.speeds_7[x_e + y_n * params.nx]
				cells_new.speeds_6[ii + jj * params.nx] = cells.speeds_8[x_w + y_n * params.nx]
				cells_new.speeds_7[ii + jj * params.nx] = cells.speeds_5[x_w + y_s * params.nx]
				cells_new.speeds_8[ii + jj * params.nx] = cells.speeds_6[x_e + y_s * params.nx]
			} else {
				local_density := float32(cells.speeds_0[ii + jj * params.nx] + cells.speeds_1[x_w + jj * params.nx] + cells.speeds_2[ii + y_s * params.nx] + cells.speeds_3[x_e + jj * params.nx] + cells.speeds_4[ii + y_n * params.nx] + cells.speeds_5[x_w + y_s * params.nx] + cells.speeds_6[x_e + y_s * params.nx] + cells.speeds_7[x_e + y_n * params.nx] + cells.speeds_8[x_w + y_n * params.nx])
				u_x := (cells.speeds_1[x_w + jj * params.nx] + cells.speeds_5[x_w + y_s * params.nx] + cells.speeds_8[x_w + y_n * params.nx] - (cells.speeds_3[x_e + jj * params.nx] + cells.speeds_6[x_e + y_s * params.nx] + cells.speeds_7[x_e + y_n * params.nx])) / local_density;
				u_y := (cells.speeds_2[ii + y_s * params.nx] + cells.speeds_5[x_w + y_s * params.nx] + cells.speeds_6[x_e + y_s * params.nx] - (cells.speeds_4[ii + y_n * params.nx] + cells.speeds_7[x_e + y_n * params.nx] + cells.speeds_8[x_w + y_n * params.nx])) / local_density;
				u_sq := u_x * u_x + u_y * u_y;
				cells_new.speeds_0[ii + jj * params.nx] = cells.speeds_0[ii + jj * params.nx] + params.omega * (w0 * local_density * (1 - u_sq * two_c_sq_r) - cells.speeds_0[ii + jj * params.nx])
				cells_new.speeds_1[ii + jj * params.nx] = cells.speeds_1[x_w + jj * params.nx] + params.omega * (w1 * local_density * (1 + u_x * c_sq_r + (u_x * u_x) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells.speeds_1[x_w + jj * params.nx])
				cells_new.speeds_2[ii + jj * params.nx] = cells.speeds_2[ii + y_s * params.nx] + params.omega * (w1 * local_density * (1 + u_y * c_sq_r + (u_y * u_y) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells.speeds_2[ii + y_s * params.nx])
				cells_new.speeds_3[ii + jj * params.nx] = cells.speeds_3[x_e + jj * params.nx] + params.omega * (w1 * local_density * (1 + -u_x * c_sq_r + (-u_x * -u_x) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells.speeds_3[x_e + jj * params.nx])
				cells_new.speeds_4[ii + jj * params.nx] = cells.speeds_4[ii + y_n * params.nx] + params.omega * (w1 * local_density * (1 + -u_y * c_sq_r + (-u_y * -u_y) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells.speeds_4[ii + y_n * params.nx])
				cells_new.speeds_5[ii + jj * params.nx] = cells.speeds_5[x_w + y_s * params.nx] + params.omega * (w2 * local_density * (1 + (u_x + u_y) * c_sq_r + ((u_x + u_y) * (u_x + u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells.speeds_5[x_w + y_s * params.nx])
				cells_new.speeds_6[ii + jj * params.nx] = cells.speeds_6[x_e + y_s * params.nx] + params.omega * (w2 * local_density * (1 + (-u_x + u_y) * c_sq_r + ((-u_x + u_y) * (-u_x + u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells.speeds_6[x_e + y_s * params.nx])
				cells_new.speeds_7[ii + jj * params.nx] = cells.speeds_7[x_e + y_n * params.nx] + params.omega * (w2 * local_density * (1 + (-u_x - u_y) * c_sq_r + ((-u_x - u_y) * (-u_x - u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells.speeds_7[x_e + y_n * params.nx])
				cells_new.speeds_8[ii + jj * params.nx] = cells.speeds_8[x_w + y_n * params.nx] + params.omega * (w2 * local_density * (1 + (u_x - u_y) * c_sq_r + ((u_x - u_y) * (u_x - u_y)) * two_c_sq_sq_r - u_sq * two_c_sq_r) - cells.speeds_8[x_w + y_n * params.nx])
				tot_u += float32(math.Sqrt(float64(u_sq)))
			}
			
	  	}
	}
	return tot_u * params.num_non_obstacles_r;
}

func swap(x *t_speed, y *t_speed) {
	temp := *x
	*x = *y
	*y = temp
}

func av_velocity(params t_param, cells t_speed, obstacles []int) float32 {
	tot_cells := 0
	tot_u := float32(0.0)

	for jj := 0; jj < params.ny; jj++ {
		for ii := 0; ii < params.nx; ii++ {
			if obstacles[ii + jj * params.nx] == 0 {
				local_density := float32(cells.speeds_0[ii + jj * params.nx] + cells.speeds_1[ii + jj * params.nx] + cells.speeds_2[ii + jj * params.nx] + cells.speeds_3[ii + jj * params.nx] + cells.speeds_4[ii + jj * params.nx] + cells.speeds_5[ii + jj * params.nx] + cells.speeds_6[ii + jj * params.nx] + cells.speeds_7[ii + jj * params.nx] + cells.speeds_8[ii + jj * params.nx])
				u_x := (cells.speeds_1[ii + jj * params.nx] + cells.speeds_5[ii + jj * params.nx] + cells.speeds_8[ii + jj * params.nx] - (cells.speeds_3[ii + jj * params.nx] + cells.speeds_6[ii + jj * params.nx] + cells.speeds_7[ii + jj * params.nx])) / local_density
				u_y := (cells.speeds_2[ii + jj * params.nx] + cells.speeds_5[ii + jj * params.nx] + cells.speeds_6[ii + jj * params.nx] - (cells.speeds_4[ii + jj * params.nx] + cells.speeds_7[ii + jj * params.nx] + cells.speeds_8[ii + jj * params.nx])) / local_density
				u_sq := u_x * u_x + u_y * u_y;
				tot_u += float32(math.Sqrt(float64(u_sq)))
				tot_cells ++
			}
		}
	}
	return tot_u / float32(tot_cells)
}

func calc_reynolds(params t_param, cells t_speed, obstacles []int) float32{
	viscosity := 1.0 / 6.0 * (2.0 / params.omega - 1.0);
	return av_velocity(params, cells, obstacles) * float32(params.reynolds_dim) / viscosity;
}

func write_values(params t_param, cells t_speed, obstacles []int, av_vels []float32) {
	c_sq := float32(1.0 / 3.0)
	file_final_state, _ := os.Create("final_state.dat")
	writer_final_state := bufio.NewWriter(file_final_state)

	for jj := 0; jj < params.ny; jj ++ {
		for ii := 0; ii < params.nx; ii ++ {
			var u_x float32
			var u_y float32
			var u float32
			var pressure float32
			if obstacles[ii + jj * params.nx] == 1 {
				u_x = 0
				u_y = 0
				u = 0
				pressure = params.density * c_sq
			} else {
				local_density := float32(cells.speeds_0[ii + jj * params.nx] + cells.speeds_1[ii + jj * params.nx] + cells.speeds_2[ii + jj * params.nx] + cells.speeds_3[ii + jj * params.nx] + cells.speeds_4[ii + jj * params.nx] + cells.speeds_5[ii + jj * params.nx] + cells.speeds_6[ii + jj * params.nx] + cells.speeds_7[ii + jj * params.nx] + cells.speeds_8[ii + jj * params.nx])
				u_x = (cells.speeds_1[ii + jj * params.nx] + cells.speeds_5[ii + jj * params.nx] + cells.speeds_8[ii + jj * params.nx] - (cells.speeds_3[ii + jj * params.nx] + cells.speeds_6[ii + jj * params.nx] + cells.speeds_7[ii + jj * params.nx])) / local_density
				u_y = (cells.speeds_2[ii + jj * params.nx] + cells.speeds_5[ii + jj * params.nx] + cells.speeds_6[ii + jj * params.nx] - (cells.speeds_4[ii + jj * params.nx] + cells.speeds_7[ii + jj * params.nx] + cells.speeds_8[ii + jj * params.nx])) / local_density
				u = float32(math.Sqrt(float64(u_x * u_x + u_y * u_y)))
				pressure = local_density * c_sq;
			}
			_, _ = writer_final_state.WriteString(fmt.Sprintf("%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]))
		}
	}

	writer_final_state.Flush()
	file_final_state.Close()

	file_av_vels, _ := os.Create("av_vels.dat")
	writer_av_vels := bufio.NewWriter(file_av_vels)

	for ii := 0; ii < params.maxIters; ii++ {
		_, _ = writer_av_vels.WriteString(fmt.Sprintf("%d:\t%.12E\n", ii, av_vels[ii]))
	}

	writer_av_vels.Flush()
	file_av_vels.Close()

}

func main() {
	paramfile := os.Args[1]
	obstaclefile := os.Args[2]

	tot_tic := time.Now()
	init_tic := tot_tic
	cells, cells_new, obstacles, params := initialise(paramfile, obstaclefile)
	av_vels := make([]float32, params.maxIters)
	init_toc := time.Now()
	comp_tic := init_toc

	for tt := 0; tt < params.maxIters; tt++ {
	// for tt := 0; tt < 2; tt++ {
		accelerate_flow(params, &cells, obstacles)
		av_vels[tt] = timestep(params, &cells, &cells_new, obstacles)
		swap(&cells, &cells_new)
	}

	comp_toc := time.Now()
	col_tic := comp_toc
	col_toc := time.Now()
	tot_toc := col_toc

	fmt.Print("==done==\n");
	fmt.Print("Reynolds number:\t\t", calc_reynolds(params, cells, obstacles), "\n");
	fmt.Print("Elapsed Init time:\t\t\t", init_toc.Sub(init_tic), "(s)\n")
	fmt.Print("Elapsed Compute time:\t\t\t", comp_toc.Sub(comp_tic), "(s)\n" )
	fmt.Print("Elapsed Collate time:\t\t\t", col_toc.Sub(col_tic), "(s)\n")
	fmt.Print("Elapsed Total time:\t\t\t", tot_toc.Sub(tot_tic), "(s)\n")

	write_values(params, cells, obstacles, av_vels)
}
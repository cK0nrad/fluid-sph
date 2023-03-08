use rayon::prelude::*;
use std::{
    f64::consts::PI,
    ops::{Add, Div, Mul, Sub},
};

use crate::vectors::Vector;

pub struct SPH {
    epsilon: f64,
    pub mass: f64,

    g: Vector,
    rest_density: f64,
    pdist: f64,
    pub pradi: f64,

    pub h: f64,
    acc_limit: f64,
    damping: f64,
    bound_repul: f64,

    kp: f64,
    visc: f64,
    tension: f64,
    dt: f64,

    wc: f64,
    grad_w_2_c: f64,

    pub positions: Vec<Vector>,
    pub velocities: Vec<Vector>,
    pub accelerations: Vec<Vector>,
    pub densities: Vec<f64>,
    grid: Vec<Vec<Vec<Vec<usize>>>>,

    len: Vector,
    bounds: Vector,
}

impl SPH {
    pub fn new(bounds: Vector, dt: f64) -> Self {
        let epsilon: f64 = 1e-4;
        let mass: f64 = 0.00020543;

        let g = Vector::new(0.0, 0.0, -9.82_f64.div(0.004));
        let rest_density: f64 = 600.0_f64.mul(0.004_f64.powi(3));
        let pdist: f64 = (mass.div(rest_density)).powf(1.0 / 3.0);
        let pradi: f64 = 0.1;

        let h: f64 = 0.01_f64.div(0.004); // kernel radius
        let acc_limit: f64 = 20000.0;
        let damping: f64 = 256.0;
        let bound_repul: f64 = 10000.0;

        let kp: f64 = 3.0_f64.div(0.004_f64.powi(2)); // Pressure Stiffness
        let visc: f64 = 0.25.mul(0.004); // Viscosity
        let tension: f64 = 150.0; // Surface Tension

        let wc: f64 = (315.0_f64).div(64.0).div(PI).div(h.powi(9));
        let grad_w_2_c: f64 = 45.0_f64.div(PI).div(h.powi(6));

        let positions = Vec::<Vector>::new();
        let velocities = Vec::<Vector>::new();
        let accelerations = Vec::<Vector>::new();

        let grid: Vec<Vec<Vec<Vec<usize>>>> = vec![vec![vec![Vec::<usize>::new(); 110]; 110]; 110];

        let len = Vector::new(
            bounds.get_x().div(100.0_f64).max(h),
            bounds.get_y().div(100.0_f64).max(h),
            bounds.get_z().div(100.0_f64).max(h),
        );

        Self {
            epsilon,
            mass,
            g,
            rest_density,
            pdist,
            pradi,
            h,
            acc_limit,
            damping,
            bound_repul,
            kp,
            visc,
            tension,
            dt,
            wc,
            grad_w_2_c,
            positions,
            velocities,
            accelerations,
            grid,
            len,
            densities: Vec::<f64>::new(),
            bounds,
        }
    }

    fn w(r: Vector, h: f64, wc: f64) -> f64 {
        let distance = r.square_size();
        let h2 = h.powi(2);

        if h2 < distance {
            return 0.0;
        }

        wc * (h2 - distance).powi(3)
    }

    fn grad_w_2(r: Vector, h: f64, grad_w_2_c: f64) -> Vector {
        let distance = r.square_size();
        let h2 = h.powi(2);
        if h2 < distance {
            return Vector::new(0.0, 0.0, 0.0);
        }

        let distance = distance.sqrt();

        let constant = -1.0 * grad_w_2_c * (h - distance).powi(2) / distance;
        Vector::new(
            constant * r.get_x(),
            constant * r.get_y(),
            constant * r.get_z(),
        )
    }

    fn laplacian_w_2(r: Vector, h: f64, grad_w_2_c: f64) -> f64 {
        let distance = r.square_size().sqrt();
        if h < distance {
            return 0.0;
        }
        grad_w_2_c * (h - distance)
    }

    pub fn add_particle(&mut self, from: &Vector, to: &Vector) {
        let epsilon = self.epsilon;
        let d = self.pdist * 0.84;
        let positions = &mut self.positions;
        let velocities = &mut self.velocities;
        let accelerations = &mut self.accelerations;
        let densities = &mut self.densities;

        let mut x = from.get_x().add(epsilon);
        let mut y = from.get_y().add(epsilon);
        let mut z = from.get_z().add(epsilon);

        while x <= to.get_x().sub(epsilon) {
            while y <= to.get_y().sub(epsilon) {
                while z <= to.get_z().sub(epsilon) {
                    positions.push(Vector::new(x, y, z));
                    velocities.push(Vector::new(0.0, 0.0, 0.0));
                    accelerations.push(Vector::new(0.0, 0.0, 0.0));
                    densities.push(0.0);
                    z += d;
                }
                y += d;
                z = from.get_y().add(epsilon);
            }
            x += d;
            y = from.get_y().add(epsilon);
        }
    }

    pub fn construct_grid(&mut self) {
        let position = &self.positions;

        for i in 0..position.len() {
            let grid_x = position[i].get_x().div(self.len.get_x()) as usize;
            let grid_y = position[i].get_y().div(self.len.get_y()) as usize;
            let grid_z = position[i].get_z().div(self.len.get_z()) as usize;
            if grid_x > 110 || grid_y > 110 || grid_z > 110 {
                println!(
                    "{} {} {}",
                    position[i].get_z().div(self.len.get_z()),
                    self.len.get_x(),
                    grid_z
                );
            }
            self.grid[grid_x][grid_y][grid_z].push(i);
        }
    }

    pub fn density(&mut self) {
        let positions = &self.positions;
        let m = &self.mass;
        let rho = &mut self.densities;

        (0..positions.len())
            .into_par_iter()
            .zip_eq(rho)
            .for_each(|(i, rho)| {
                let grid_x = positions[i].get_x().div(self.len.get_x()) as i32;
                let grid_y = positions[i].get_y().div(self.len.get_y()) as i32;
                let grid_z = positions[i].get_z().div(self.len.get_z()) as i32;

                *rho = 0.0;
                for x in 0..3_i32 {
                    if x.sub(1).add(grid_x) < 0 {
                        continue;
                    }
                    for y in 0..3_i32 {
                        if y.sub(1).add(grid_y) < 0 {
                            continue;
                        }
                        for z in 0..3_i32 {
                            if z.sub(1).add(grid_z) < 0 {
                                continue;
                            }

                            for j in &self.grid[grid_x.sub(1).add(x) as usize]
                                [grid_y.sub(1).add(y) as usize]
                                [grid_z.sub(1).add(z) as usize]
                            {
                                let direction = positions[*j].subv(positions[i]);
                                *rho += Self::w(direction, self.h, self.wc);
                            }
                        }
                    }
                }
                *rho *= m;
            });
    }

    pub fn pressure(rho: f64, kp: f64, rho0: f64) -> f64 {
        kp * (rho - rho0)
    }

    pub fn accelerate(&mut self) {
        (0..self.positions.len())
            .into_par_iter()
            .zip_eq(&mut self.accelerations)
            .for_each(|(i, accel)| {
                let f_gravity = self.g;

                let mut f_tens = Vector::new(0.0, 0.0, 0.0);
                let mut f_pres = Vector::new(0.0, 0.0, 0.0);
                let mut f_visc = Vector::new(0.0, 0.0, 0.0);

                let grid_x = self.positions[i].get_x().div(self.len.get_x()) as i32;
                let grid_y = self.positions[i].get_y().div(self.len.get_y()) as i32;
                let grid_z = self.positions[i].get_z().div(self.len.get_z()) as i32;

                for x in 0..3_i32 {
                    if x.sub(1).add(grid_x) < 0 {
                        continue;
                    }
                    for y in 0..3_i32 {
                        if y.sub(1).add(grid_y) < 0 {
                            continue;
                        }
                        for z in 0..3_i32 {
                            if z.sub(1).add(grid_z) < 0 {
                                continue;
                            }

                            for j in &self.grid[grid_x.add(x).sub(1) as usize]
                                [grid_y.add(y).sub(1) as usize]
                                [grid_z.add(z).sub(1) as usize]
                            {
                                if i == *j {
                                    continue;
                                }

                                let direction = self.positions[i].subv(self.positions[*j]);

                                let press =
                                    Self::pressure(self.densities[i], self.kp, self.rest_density)
                                        .add(Self::pressure(
                                            self.densities[*j],
                                            self.kp,
                                            self.rest_density,
                                        ))
                                        .div(2.0);

                                let tension = Self::w(direction, self.h, self.wc)
                                    .mul(self.densities[i])
                                    .mul(self.tension);

                                f_tens = f_tens.subv(direction.mulf(tension));

                                let pression = press.mul(self.mass).div(self.densities[*j]);
                                f_pres = f_pres.subv(
                                    Self::grad_w_2(direction, self.h, self.grad_w_2_c)
                                        .mulf(pression),
                                );

                                let viscosity =
                                    Self::laplacian_w_2(direction, self.h, self.grad_w_2_c)
                                        .mul(self.visc)
                                        .mul(self.mass)
                                        .div(self.densities[*j]);

                                f_visc = f_visc.addv(
                                    self.velocities[*j].subv(self.velocities[i]).mulf(viscosity),
                                );
                                if x == 0 && y == 0 {}
                            }
                        }
                    }
                }
                let f = f_tens.addv(f_pres).addv(f_visc);
                *accel = f.divf(self.densities[i]).addv(f_gravity);
            });
    }

    pub fn update_position(&mut self) {
        let bounds = self.bounds;

        for x in 0..110 {
            for y in 0..110 {
                for z in 0..110 {
                    self.grid[x][y][z].clear();
                }
            }
        }

        (0..self.positions.len())
            .into_par_iter()
            .rev()
            .zip_eq(&mut self.positions)
            .zip_eq(&mut self.velocities)
            .zip_eq(&mut self.accelerations)
            .for_each(|(((_, position), velocity), acceleration)| {

                if acceleration.get_x().is_nan() {
                    acceleration.set_x(self.acc_limit);
                }
                if acceleration.get_y().is_nan() {
                    acceleration.set_y(self.acc_limit);
                }
                if acceleration.get_z().is_nan() {
                    acceleration.set_z(self.acc_limit);
                }

                let accel = acceleration.square_size();
                if accel > self.acc_limit.powi(2) {
                    *acceleration = acceleration
                        .mulf(self.acc_limit)
                        .divf(accel.sqrt());
                }

                let mut normal_x = 0.0;
                let mut normal_y = 0.0;
                let mut normal_z = 0.0;

                let mut xdisp = 0.0;
                let mut ydisp = 0.0;
                let mut zdisp = 0.0;

                if position.get_x() < self.pradi {
                    normal_x = 1.0;
                    xdisp = self.pradi - position.get_x();
                    position.set_x(self.pradi);
                }
                if (bounds.get_x() - position.get_x()) < self.pradi {
                    normal_x = -1.0;
                    xdisp = self.pradi - (bounds.get_x() - position.get_x());
                    position.set_x(bounds.get_x().sub(self.pradi));
                }

                if position.get_y() < self.pradi {
                    normal_y = 1.0;
                    ydisp = self.pradi - position.get_y();
                    position.set_y(self.pradi);
                }
                if (bounds.get_y() - position.get_y()) < self.pradi {
                    normal_y = -1.0;
                    ydisp = self.pradi - (bounds.get_y() - position.get_y());
                    position.set_y(bounds.get_z().sub(self.pradi));
                }

                if position.get_z() < self.pradi {
                    normal_z = 1.0;
                    zdisp = self.pradi - position.get_z();
                    position.set_z(self.pradi);
                }
                if (bounds.get_z() - position.get_z()) < self.pradi {
                    normal_z = -1.0;
                    zdisp = self.pradi - (bounds.get_z() - position.get_z());
                    position.set_z(bounds.get_z().sub(self.pradi));
                }

                let normal = Vector::new(normal_x, normal_y, normal_z);

                let dot_product = velocity.dot(normal);

                let x_acceleration =
                    self.bound_repul * xdisp * normal_x - self.damping * dot_product * normal_x;
                let y_acceleration =
                    self.bound_repul * ydisp * normal_y - self.damping * dot_product * normal_y;
                let z_acceleration =
                    self.bound_repul * zdisp * normal_z - self.damping * dot_product * normal_z;

                let acceleration_vec = Vector::new(x_acceleration, y_acceleration, z_acceleration);

                *acceleration = acceleration.addv(acceleration_vec);
                *velocity = velocity.addv(acceleration.mulf(self.dt));
                *position = position.addv(velocity.mulf(self.dt));
            });

        self.construct_grid();
    }
}

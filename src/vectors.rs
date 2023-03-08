use serde::{Serialize, Deserialize};

 impl Vector {
    pub  fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn set(&mut self, i: usize, value: f64) {
        match i {
            0 => self.set_x(value),
            1 => self.set_y(value),
            2 => self.set_z(value),
            _ => {}
        }
    }

    pub fn get(&self, i:usize) -> f64 {
        match i {
            0 => self.get_x(),
            1 => self.get_y(),
            2 => self.get_z(),
            _ => 0.0
        }
    }
    pub fn get_x(&self) -> f64 {
        return self.x;
    }

    pub fn get_y(&self) -> f64 {
        return self.y;
    }

    pub fn get_z(&self) -> f64 {
        return self.z;
    }

    pub fn set_x(&mut self, i: f64){
        self.x = i;
    }

    pub fn set_y(&mut self, i: f64){
        self.y = i;
    }

    pub fn set_z(&mut self, i: f64){
        self.z = i;
    }

    // pub fn shift(&mut self, acc:Self, dt: f64){
    //     self.x += dt*acc.get_x();
    //     self.y += dt*acc.get_y();
    //     self.z += dt*acc.get_z();
    // }

    pub fn dot(&self, other: Self) -> f64{
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    // pub fn opposite_x(&mut self, dampening: f64){
    //     self.x = -self.x * dampening;
    // }

    // pub fn opposite_y(&mut self, dampening: f64){
    //     self.y = -self.y * dampening;
    // }
    
    // pub fn opposite_z(&mut self, dampening: f64){
    //     self.z = -self.z * dampening;
    // }

    pub fn square_size(&self) -> f64 {
        return self.x.powi(2) + self.y.powi(2) + self.z.powi(2);
    }
    //Math operation: 

    pub fn subv(&self, other: Self) -> Self  {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z
        }
    }

    pub fn subf(&self, other: f64) -> Self  {
        Self {
            x: self.x - other,
            y: self.y - other,
            z: self.z - other
        }
    }

    pub fn addv(&self, other: Self) -> Self  {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z
        }
    }

    pub fn addf(&self, other: f64) -> Self  {
        Self {
            x: self.x + other,
            y: self.y + other,
            z: self.z + other
        }
    }


    pub fn mulf(&self, other: f64) -> Self {
        Self {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other
        }
    }

    pub fn mulv(&self, other: Self) -> Self {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z
        }
    }

    pub fn divf(&self, other: f64) -> Self {
        Self {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other
        }
    }

}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Vector {
    x: f64,
    y: f64,
    z: f64
}
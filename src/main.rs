mod vectors;
mod sph;
mod pbrt;
mod eigen_value;
//https://elrnv.com/cs888/cs888proj.pdf
use std::time::Instant;

use prgrs::Prgrs;
use vectors::Vector;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DensityPosition {
    pub vector: Vector,
    pub density: f64,
    pub timestamp: f64
}

// static TOTAL: usize = 500;
static TIME: usize = 50;
static DT: f64 = 0.004;//1.0/144.0;

impl DensityPosition {
    pub fn new(vector: Vector, density: f64, timestamp: f64) -> Self{
        Self{
            vector,
            density,
            timestamp
        }
    }
}

use three_d::*;

use crate::sph::SPH;

pub fn main() {
    let bounds = Vector::new(30.0, 30.0, 30.0);
    let from = Vector::new(0.0,0.0,0.0);
    let to = Vector::new(10.0, 10.0, 10.0);

    let mut sph = SPH::new(bounds, DT);
    sph.add_particle(&from, &to);
    sph.construct_grid();

    let mut t:f64 = 0.0;
    let mut end:Vec<DensityPosition>= Vec::new();
    
    //start tilme
    let start = Instant::now();
    

    for _ in Prgrs::new(0..TIME, TIME) {

        sph.density();

        for i in 0..sph.positions.len(){
            end.push(
                DensityPosition::new(sph.positions[i].clone(), sph.densities[i], t)
            )
        }

        sph.accelerate();
        sph.update_position();

        
        t+=DT;
    }
    
    //Get time took 
    let duration = start.elapsed();
    println!("Time took: {:?}", duration);

    let mut prbt = pbrt::Renderer::new(sph.positions.len(), 0, end, sph.h, sph.mass);

    for i in Prgrs::new(0..TIME, TIME){
        prbt.set_frame(i);
        prbt.generate();
    }

/*
    let window = Window::new(WindowSettings {
        title: "Shapes!".to_string(),
        max_size: Some((1280, 720)),
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();

    let mut camera = Camera::new_perspective(
        window.viewport(),
        vec3(-5.0, -5.0, 5.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 0.0, 1.0),
        degrees(45.0),
        0.1,
        1000.0,
    );

    let mut control = OrbitControl::new(*camera.target(), 1.0, 100.0);
    let mut spheres: Vec<Gm<Mesh, PhysicalMaterial>> = Vec::new();
    for _ in 0..sph.positions.len() {
        spheres.push(
        Gm::new(
        Mesh::new(&context, &CpuMesh::sphere(16)),
        PhysicalMaterial::new_opaque(
            &context,
            &CpuMaterial {
                albedo: Color {
                    r: 255,
                    g: 255,
                    b: 0,
                    a: 200,
                },
                ..Default::default()
            },
        ),
        ));
    }

    let mut boxe = Vec::new();
    

    //Bottom
    let mut face = Gm::new(
        Mesh::new(&context, &CpuMesh::square()),
        PhysicalMaterial::new_opaque(
            &context,
            &CpuMaterial {
                albedo: Color {
                    r: 0,
                    g: 0,
                    b: 255,
                    a: 100,
                },
                roughness: 0.4,
                ..Default::default()
            },
        ),
    );
    face.set_transformation(
        Mat4::from_translation(
            vec3(
                (0.0) as f32, 
                (0.0) as f32, 
                (0.0) as f32 
            )
        ) * Mat4::from_scale(5.0)
    );
    boxe.push(face);
    
    
    let mut last_update = Instant::now();
    let mut delay = 1.0/144.0;
    let mut paused = false;
    let light0 = DirectionalLight::new(&context, 1.0, Color::WHITE, &vec3(1.0, 1.0, -1.0));
    let mut k = 0;
    window.render_loop(move |mut frame_input: FrameInput| {
        for event in &frame_input.events {
            match event {
                Event::KeyPress { kind, modifiers: _, handled: _ } => {
                    if *kind == Key::R {
                        k = 0;
                    }

                    if *kind == Key::Space {
                        paused = !paused;
                    }

                    if *kind == Key::Z {
                        delay += 0.0005;
                        println!("New delay: {}", delay);
                    }

                    if *kind == Key::S {
                        delay -= 0.0005;
                        if delay < 0.0 {
                            delay = 0.0;
                        } 
                        println!("New delay: {}", delay);
                    }

                    if *kind == Key::E {
                        delay = 1.0/144.0;
                        println!("New delay: {}", delay);
                    }
                },
                _ => {}
            }
        }
        

        camera.set_viewport(frame_input.viewport);
        control.handle_events(&mut camera, &mut frame_input.events);

        if k >= end.len() {
            k = 0
        } 
        for i in 0..sph.positions.len() {
            let current = end[k+i];
            spheres[i].set_transformation(
                Mat4::from_translation(
                    vec3(
                        current.vector.get_x() as f32, 
                        current.vector.get_y() as f32, 
                        current.vector.get_z() as f32 
                    )
                ) * Mat4::from_scale(0.1)
            );
        }


        frame_input
            .screen()
            .clear(ClearState::color_and_depth(0.8, 0.8, 0.8, 1.0, 1.0))
            .render(
                &camera,
                spheres.iter()
                    .into_iter().chain(&boxe),
                &[&light0],
            );
        // thread::sleep(time::Duration::from_millis((DT) as u64));
        if !paused && last_update.elapsed().as_secs_f64() > delay {
            k += sph.positions.len();
            last_update = Instant::now();
        }
        
        FrameOutput::default()
    });  */
} 
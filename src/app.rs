use std::{
    iter,
    sync::{Arc, Mutex},
};

use eframe::{
    App,
    egui::{self, Context, TextureHandle, Vec2},
};

use nalgebra as na;

use crate::{
    RayModel,
    gpu::{self, GPUHandle},
};

#[derive(PartialEq, Eq)]
enum MyAppState {
    Idle,
    NeedsUpdate,
    Updating,
}

struct PosState {
    cur: f64,
    prev: f64,
    text: String,
    step: f64,
    step_text: String,
}

pub struct MyApp {
    xpos: PosState,
    ypos: PosState,
    zpos: PosState,
    //pipeline: wgpu::ComputePipeline,
    texture_handle: egui::TextureHandle,
    width: usize,
    height: usize,
    max_iters: usize,
    state: Arc<Mutex<MyAppState>>,
    render_state: eframe::egui_wgpu::RenderState,
    gpu_handle: Arc<Mutex<gpu::GPUHandle>>,
}

impl MyApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // let width = 40 * 16;
        // let height = 24 * 16;

        let width = 80 * 16;
        let height = 48 * 16;

        let wgpu_render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("WGPU render state is required for compute shaders");

        let device = &wgpu_render_state.device;

        let (
            input_buffer,
            output_buffer,
            gpu_vars_buffer,
            output_staging_buffer,
            compute_pipeline,
            bind_group,
        ) = gpu::setup_compute(device, width, height);

        // Initialize your compute pipeline here...
        // let pipeline = device.create_compute_pipeline(...);

        let pixels: Vec<_> = iter::repeat_n(0u8, 4 * width * height).collect();
        let color_image = egui::ColorImage::from_rgba_unmultiplied([width, height], &pixels);

        let texture_handle =
            cc.egui_ctx
                .load_texture("test texture", color_image, egui::TextureOptions::NEAREST);

        let gpu_handle = Arc::new(Mutex::new(gpu::GPUHandle {
            input_buffer,
            output_buffer,
            gpu_vars_buffer,
            output_staging_buffer,
            compute_pipeline,
            bind_group,
        }));

        let state = Arc::new(Mutex::new(MyAppState::NeedsUpdate));

        Self {
            xpos: PosState {
                cur: 0.0,
                prev: 0.0,
                text: "0.0".to_owned(),
                step: 0.05,
                step_text: "0.05".to_owned(),
            },
            ypos: PosState {
                cur: 0.0,
                prev: 0.0,
                text: "0.0".to_owned(),
                step: 0.05,
                step_text: "0.05".to_owned(),
            },
            zpos: PosState {
                cur: -4.0,
                prev: -4.0,
                text: "0.0".to_owned(),
                step: 0.05,
                step_text: "0.05".to_owned(),
            },
            texture_handle,
            width,
            height,
            max_iters: 1000,
            state,
            gpu_handle,
            render_state: wgpu_render_state.clone(),
        }
    }
}

impl App for MyApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let Self {
            xpos,
            ypos,
            zpos,
            texture_handle,
            state,
            width,
            height,
            ..
        } = self;

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.set_max_width(300.0);

                    // The central panel the region left after adding TopPanel's and SidePanel's
                    ui.heading("Mandelbulb DE");

                    macro_rules! set_pos {
                        ($label:expr, $pos:expr) => {
                            ui.horizontal(|ui| {
                                ui.label($label);
                                let response = ui.add_enabled(
                                    *state.lock().unwrap() == MyAppState::Idle,
                                    egui::TextEdit::singleline(&mut $pos.text).desired_width(50.0),
                                );
                                if response.lost_focus() {
                                    if let Ok(num) = $pos.text.parse::<f64>() {
                                        $pos.cur = num;
                                        *state.lock().unwrap() = MyAppState::NeedsUpdate
                                    }
                                }

                                ui.label("Step Size:");
                                let response = ui.add_enabled(
                                    *state.lock().unwrap() == MyAppState::Idle,
                                    egui::TextEdit::singleline(&mut $pos.step_text),
                                );
                                if response.changed() {
                                    if let Ok(num) = $pos.step_text.parse::<f64>() {
                                        $pos.step = num;
                                        $pos.prev = $pos.cur;
                                    }
                                }
                            });
                            let slider_response = ui.add_enabled(
                                *state.lock().unwrap() == MyAppState::Idle,
                                egui::Slider::new(
                                    &mut $pos.cur,
                                    $pos.prev - ($pos.step * 50.0)..=$pos.prev + ($pos.step * 50.0),
                                )
                                .text("x position"),
                            );
                            if slider_response.changed() {
                                $pos.text = $pos.cur.to_string();
                                *state.lock().unwrap() = MyAppState::NeedsUpdate
                            }
                        };
                    }

                    set_pos!("Camera X:", xpos);
                    set_pos!("Camera Y:", ypos);
                    set_pos!("Camera Z:", zpos);
                });

                let avail_width = ui.available_width();
                let size = Vec2::new(
                    avail_width,
                    (*height as f32) / (*width as f32) * avail_width,
                );

                let sized_texture = egui::load::SizedTexture::new(texture_handle.id(), size);
                ui.image(sized_texture);
            });
        });
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if ctx.input(|i| {
            i.events.iter().any(|e| {
                matches!(
                    e,
                    egui::Event::Key {
                        key: egui::Key::Space,
                        pressed: true,
                        ..
                    }
                )
            })
        }) {
            //self.xpos += 0.1;
            *self.state.lock().unwrap() = MyAppState::NeedsUpdate;
        }

        if *self.state.lock().unwrap() == MyAppState::NeedsUpdate {
            *self.state.lock().unwrap() = MyAppState::Updating;
            run_update(
                self.xpos.cur,
                self.ypos.cur,
                self.zpos.cur,
                self.width,
                self.height,
                self.max_iters as u32,
                ctx.clone(),
                self.texture_handle.clone(),
                self.gpu_handle.clone(),
                self.render_state.clone(),
                self.state.clone(),
            );
        }
    }
}

fn run_update(
    xpos: f64,
    ypos: f64,
    zpos: f64,
    width: usize,
    height: usize,
    max_iters: u32,
    ctx: Context,
    mut texture_handle: TextureHandle,
    gpu_handle: Arc<Mutex<GPUHandle>>,
    render_state: eframe::egui_wgpu::RenderState,
    state: Arc<Mutex<MyAppState>>,
) {
    wasm_bindgen_futures::spawn_local(async move {
        let image = update_texture(
            xpos,
            ypos,
            zpos,
            gpu_handle.clone(),
            render_state.clone(),
            width,
            height,
            max_iters,
        )
        .await;
        texture_handle.set(image, egui::TextureOptions::NEAREST);
        *state.lock().unwrap() = MyAppState::Idle;
        ctx.request_repaint();
    });
}

async fn update_texture(
    xpos: f64,
    ypos: f64,
    zpos: f64,
    gpu_handle: Arc<Mutex<GPUHandle>>,
    render_state: eframe::egui_wgpu::RenderState,
    width: usize,
    height: usize,
    max_iters: u32,
) -> egui::ColorImage {
    let mut pixels: Vec<_> = iter::repeat_n(0u8, 4 * width * height).collect();

    let ray_model = RayModel::new(
        na::Point3::new(xpos, ypos, zpos),
        na::Point3::new(0.0, 0.0, 0.0),
        na::Vector3::y(),
        16.0 / 9.0,
        3.14 / 4.0,
        1.0,
        10000.0,
    );

    let origin = ray_model.origin;

    let rays = {
        let mut result = vec![];

        for y in 0..height {
            for x in 0..width {
                let pout = ray_model.proj_for_screen_xy(x, y, width, height);
                let dir = pout - origin;
                let ray = gpu::RayInput::new(
                    [pout.x as f32, pout.y as f32, pout.z as f32],
                    [dir.x as f32, dir.y as f32, dir.z as f32],
                );
                result.push(ray);
            }
        }

        result
    };

    let gpu_vars = gpu::GPUVars {
        width: width as u32,
        height: height as u32,
        max_iterations: max_iters,
    };

    let ray_outputs = {
        let gpu = gpu_handle.try_lock();

        if let Ok(gpu) = gpu {
            let rays = gpu::execute(
                &render_state.queue,
                &render_state.device,
                &gpu_vars,
                &gpu.gpu_vars_buffer,
                &gpu.input_buffer,
                &gpu.output_buffer,
                &gpu.output_staging_buffer,
                &gpu.compute_pipeline,
                &gpu.bind_group,
                rays,
                width,
                height,
            )
            .await;
            Some(rays)
        } else {
            None
        }
    };

    if let Some(ray_outputs) = ray_outputs {
        let mut i = 0;
        for y in 0..height {
            for x in 0..width {
                let ray = ray_outputs[i];

                let val = if ray.did_hit == 0 {
                    (0, 0, 0)
                } else {
                    let hsv = egui::ecolor::rgb_from_hsv((
                        ray.iters as f32 / max_iters as f32,
                        0.75,
                        0.75,
                    ));

                    (
                        (hsv[0] * 256.0).floor() as u8,
                        (hsv[1] * 256.0).floor() as u8,
                        (hsv[2] * 256.0).floor() as u8,
                    )
                };

                let idx = 4 * width * y + 4 * x;
                pixels[idx] = val.0;
                pixels[idx + 1] = val.1;
                pixels[idx + 2] = val.2;
                pixels[idx + 3] = 255;

                i += 1;
            }
        }
    }

    egui::ColorImage::from_rgba_unmultiplied([width, height], &pixels)
}

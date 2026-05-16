use nalgebra as na;

use crate::app::MyApp;

mod app;
mod gpu;
mod reference;

#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast;

    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    //let web_options = eframe::WebOptions::default();

    let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_descriptor.backends = wgpu::Backends::all();

    let mut wgpu_setup = eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    wgpu_setup.instance_descriptor = instance_descriptor;
    wgpu_setup.device_descriptor = std::sync::Arc::new(|_adapter| wgpu::DeviceDescriptor {
        label: Some("Compute Device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
        // Explicitly handle the new fields introduced in recent wgpu versions
        memory_hints: wgpu::MemoryHints::default(),
        ..Default::default()
    });

    let web_options = eframe::WebOptions {
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(wgpu_setup),
            ..Default::default()
        },
        ..Default::default()
    };

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window().unwrap().document().unwrap();

        let canvas: web_sys::HtmlCanvasElement = document
            .get_element_by_id("primary_canvas")
            .unwrap()
            .dyn_into()
            .unwrap();

        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(MyApp::new(cc)))),
            )
            .await;

        if let Some(loading_text) = document.get_element_by_id("loading_text") {
            match start_result {
                Ok(_) => {
                    loading_text.remove();
                }
                Err(e) => {
                    loading_text.set_inner_html(
                        "<p> The app has crashed. See the developer console for details. </p>",
                    );
                    panic!("Failed to start eframe: {e:?}");
                }
            }
        }
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Error> {
    env_logger::init();

    //gpu_main()?;

    let ray_model = RayModel::new(
        na::Point3::new(0.0, 0.0, -3.0),
        na::Point3::new(0.0, 0.0, 0.0),
        na::Vector3::y(),
        16.0 / 9.0,
        3.14 / 4.0,
        1.0,
        10000.0,
    );

    for y in 0..25 {
        for x in 0..100 {
            let pout = ray_model.proj_for_screen_xy(x, y, 100, 25);
            let intersection = reference::ray_march(pout, pout - ray_model.origin);
            match intersection {
                Some((_pos, iters)) if iters < 4 => print!("#"),
                Some((_pos, iters)) if iters < 5 => print!("/"),
                Some(_) => print!("."),
                None => print!(" "),
            }
        }
        println!();
    }

    Ok(())
}

#[allow(dead_code)]
#[derive(Debug)]
struct RayModel {
    pub origin: na::Point3<f64>,
    lookat: na::Point3<f64>,
    up: na::Vector3<f64>,
    aspect: f64,
    fov: f64,
    near: f64,
    far: f64,

    persp: na::Perspective3<f64>,
    camera: na::Isometry3<f64>,
    transform: na::Matrix4<f64>,
}

impl RayModel {
    fn new(
        origin: na::Point3<f64>,
        lookat: na::Point3<f64>,
        up: na::Vector3<f64>,
        aspect: f64,
        fov: f64,
        near: f64,
        far: f64,
    ) -> Self {
        let persp = na::Perspective3::new(aspect, fov, near, far);
        let camera = na::Isometry3::look_at_rh(&origin, &lookat, &up);

        let mut model = Self {
            origin,
            lookat,
            up,
            aspect,
            fov,
            near,
            far,
            persp,
            camera,
            transform: na::Matrix4::identity(),
        };

        model.update_transform();

        model
    }

    fn update_transform(&mut self) {
        self.transform = self.camera.inverse().to_matrix() * self.persp.inverse();
    }

    fn proj_for_screen_xy(
        &self,
        sx: usize,
        sy: usize,
        x_pixels: usize,
        y_pixels: usize,
    ) -> na::Point3<f64> {
        let x = (2.0 * sx as f64 / x_pixels as f64) - 1.0;
        let y = (2.0 * sy as f64 / y_pixels as f64) - 1.0;
        let z = -1.0;
        let unit_box_point = na::Point3::new(x, y, z);
        self.transform.transform_point(&unit_box_point)
    }
}

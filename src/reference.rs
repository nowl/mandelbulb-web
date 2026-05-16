use nalgebra::*;

#[allow(dead_code)]
pub fn ray_march(origin: Point3<f64>, dir: Vector3<f64>) -> Option<(Point3<f64>, usize)> {
    let epsilon = 1e-3;
    let max_iters = 255;
    let mut iters = 0;
    let mut cur_pos = origin;

    while iters < max_iters {
        let (dist, _dist_iters) = mandelbulb_de(cur_pos.coords);
        iters += 1;
        if dist <= epsilon {
            return Some((cur_pos, iters));
        }
        cur_pos += dir * dist;
    }

    None
}

#[allow(dead_code)]
fn mandelbulb_de(pos: Vector3<f64>) -> (f64, usize) {
    let bailout = 4.0;
    let power = 8.0;
    let max_iters = 1000;

    let mut z = pos;
    let mut dr = 1.0;
    let mut r = 0.0;
    let mut iters = 1;
    while iters < max_iters {
        r = z.norm();
        if r > bailout {
            break;
        }

        // to polar
        let theta = (z.z / r).acos();
        let phi = z.y.atan2(z.x);
        dr = r.powf(power - 1.0) * power * dr + 1.0;

        // transform
        let zr2 = r.powf(power);
        let theta2 = theta * power;
        let phi2 = phi * power;

        // to cartesian
        z.x = theta2.sin() * phi2.cos();
        z.y = phi2.sin() * theta2.sin();
        z.z = theta2.cos();
        z *= zr2;
        z += pos;

        iters += 1;
    }

    (0.5 * r.log10() * r / dr, iters)
}

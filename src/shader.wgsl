struct RayInput {
    origin: vec3f,
    dir: vec3f,
}
@group(0) @binding(0)
var<storage, read> inputs: array<RayInput>; // this is used as both input and output for convenience

struct RayMarchResult {
    pos: vec3f,
    did_hit: u32,
    iters: u32
}
@group(0) @binding(1)
var<storage, read_write> outputs: array<RayMarchResult>; // this is used as both input and output for convenience

struct GPUVars {
    width: u32,
    height: u32,
    max_iterations: u32,
}
@group(0) @binding(2)
var<uniform> vars_data: GPUVars;

fn mandelbulb_de(pos: vec3f) -> f32 {
    let bailout = 4.0;
    let power = 8.0;

    var z = pos;
	var dr = 1.0f;
	var r = 0.0f;
	var iters: u32 = 0;
	while (iters < vars_data.max_iterations) {
		r = length(z);
		if r > bailout {
		    break;
		}

		// to polar
		let theta = acos(z.z/r);
		let phi = atan2(z.y,z.x);
		dr = pow(r, power-1.0)*power*dr + 1.0;

		// transform
		let zr2 = pow(r,power);
		let theta2 = theta*power;
		let phi2 = phi*power;

		// to cartesian
		z = zr2*vec3(sin(theta2)*cos(phi2), sin(phi2)*sin(theta2), cos(theta2));
		z += pos;

		iters += 1;
	}

	return 0.5*log(r)*r/dr;
}

fn ray_march(origin: vec3f, dir: vec3f) -> RayMarchResult {
    let epsilon = 1e-6;
    let max_iters: u32 = vars_data.max_iterations;
    var iters: u32 = 0;
    var cur_pos = origin;
    var did_hit: u32 = 0;

    while (iters < max_iters) {
        let dist = mandelbulb_de(cur_pos);
        iters += 1;
        if dist <= epsilon {
            did_hit = 1;
            break;
        }
        cur_pos += dir * dist;
    }

    var result: RayMarchResult;
    result.did_hit = did_hit;
    result.iters = iters;
    result.pos = cur_pos;
    return result;
}

@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = global_id.y * num_workgroups.x*16 + global_id.x;
    outputs[idx] = ray_march(inputs[idx].origin, inputs[idx].dir);
}

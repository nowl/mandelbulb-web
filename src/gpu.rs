use std::borrow::Cow;
use std::time::Duration;
use wgpu::*;

pub struct GPUHandle {
    pub input_buffer: Buffer,
    pub output_buffer: Buffer,
    pub gpu_vars_buffer: Buffer,
    pub output_staging_buffer: Buffer,
    pub compute_pipeline: ComputePipeline,
    pub bind_group: BindGroup,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
pub struct GPUVars {
    pub width: u32,
    pub height: u32,
    pub max_iterations: u32,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
pub struct RayInput {
    origin: [f32; 3],
    _padding1: f32,
    dir: [f32; 3],
    _padding2: f32,
}

impl RayInput {
    pub fn new(origin: [f32; 3], dir: [f32; 3]) -> Self {
        Self {
            origin,
            _padding1: 0.0,
            dir,
            _padding2: 0.0,
        }
    }
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
pub struct RayOutput {
    pub pos: [f32; 3],
    //_padding1: f32,
    pub did_hit: u32,
    pub iters: u32,
    _padding2: [f32; 3],
}

pub async fn execute(
    queue: &Queue,
    device: &Device,
    gpu_vars: &GPUVars,
    gpu_vars_buffer: &Buffer,
    input_buffer: &Buffer,
    output_buffer: &Buffer,
    output_staging_buffer: &Buffer,
    compute_pipeline: &ComputePipeline,
    bind_group: &BindGroup,
    inputs: Vec<RayInput>,
    width: usize,
    height: usize,
) -> Vec<RayOutput> {
    let inputs = bytemuck::cast_slice(&inputs);
    queue.write_buffer(gpu_vars_buffer, 0, bytemuck::bytes_of(gpu_vars));

    queue.write_buffer(input_buffer, 0, inputs);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(compute_pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.insert_debug_marker("compute iterations");
        cpass.dispatch_workgroups(width as u32 / 16, height as u32 / 16, 1);
    }
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &output_staging_buffer,
        0,
        output_buffer.size(),
    );

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = output_staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device
        .poll(wgt::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(10)),
        })
        .expect("timeout");

    // Awaits until `buffer_future` can be read from
    if let Ok(Ok(())) = receiver.recv_async().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result: Vec<RayOutput> = bytemuck::cast_slice(&data).to_vec();
        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        output_staging_buffer.unmap(); // Unmaps buffer from memory
        // If you are familiar with C++ these 2 lines can be thought of similarly to:
        //   delete myPointer;
        //   myPointer = NULL;
        // It effectively frees the memory

        // Returns data from buffer
        result
    } else {
        panic!("failed to run compute on gpu!")
    }
}

pub fn setup_compute(
    device: &Device,
    width: usize,
    height: usize,
) -> (Buffer, Buffer, Buffer, Buffer, ComputePipeline, BindGroup) {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let input_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Input Buffer"),
        size: (std::mem::size_of::<RayInput>() * width * height) as BufferAddress,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let output_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Output Buffer"),
        size: (std::mem::size_of::<RayOutput>() * width * height) as BufferAddress,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let output_staging_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let gpu_vars_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Variables Buffer"),
        size: std::mem::size_of::<GPUVars>() as BufferAddress,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // A bind group defines how buffers are accessed by shaders.
    // It is to WebGPU what a descriptor set is to Vulkan.
    // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

    // A pipeline specifies the operation of a shader

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: gpu_vars_buffer.as_entire_binding(),
            },
        ],
    });

    (
        input_buffer,
        output_buffer,
        gpu_vars_buffer,
        output_staging_buffer,
        compute_pipeline,
        bind_group,
    )
}

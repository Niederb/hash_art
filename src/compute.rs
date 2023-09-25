use async_trait::async_trait;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use sha2::digest::Update;
use sha2::{Digest, Sha512};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

// Indicates a u32 overflow in an intermediate Collatz value
const OVERFLOW: u8 = 0xff;

#[async_trait]
pub trait BlockApproximator {
    async fn approximate(
        &self,
        input: &ArrayView2<u8>,
        target: &ArrayView2<u8>,
    ) -> (f32, Array2<u8>, Array2<u8>);
}

pub const N: usize = 16;
pub const DISTORTION: u8 = 2;

pub struct Sha512CpuApproximator {
    iterations: u64,
}

impl Sha512CpuApproximator {
    pub fn new(iterations: u64) -> Self {
        Sha512CpuApproximator { iterations }
    }
}

#[async_trait]
impl BlockApproximator for Sha512CpuApproximator {
    async fn approximate(
        &self,
        input: &ArrayView2<u8>,
        target: &ArrayView2<u8>,
    ) -> (f32, Array2<u8>, Array2<u8>) {
        let mut best_source = Array2::<u8>::zeros((N, N));
        let mut best_target = Array2::<u8>::zeros((N, N));

        let mut error = f32::MAX;

        for _ in 0..self.iterations {
            let delta: Array2<u8> = Array::random((N, N), Uniform::new(0, DISTORTION));
            let current_source = delta + input;
            let input_vec = current_source.as_slice().unwrap();
            let mut hasher = Sha512::new();
            Update::update(&mut hasher, input_vec);
            let result = hasher.finalize();

            let current_target = Array::from_iter(result).into_shape((N, N)).unwrap();

            let mut total_error = 0.0;
            for m in 0..N {
                for n in 0..N {
                    let val = target[[m, n]] as f32 - current_target[[m, n]] as f32;
                    total_error += val * val;
                }
            }

            if error > total_error {
                best_source = current_source;
                best_target = current_target;
                error = total_error;
            }
        }
        (error, best_source, best_target)
    }
}

pub struct WgpuApproximator {}

impl WgpuApproximator {
    pub fn new() -> Self {
        WgpuApproximator {}
    }
}

#[async_trait]
impl BlockApproximator for WgpuApproximator {
    async fn approximate(
        &self,
        input: &ArrayView2<u8>,
        target: &ArrayView2<u8>,
    ) -> (f32, Array2<u8>, Array2<u8>) {
        let vals = self.run(input.to_slice().unwrap()).await;
        let result = Array2::from_shape_vec((N, N), vals).unwrap();
        (0.0, input.to_owned(), result)
    }
}

impl WgpuApproximator {
    pub async fn run(&self, numbers: &[u8]) -> Vec<u8> {
        let steps = self.execute_gpu(numbers).await.unwrap();

        steps
            .iter()
            .map(|&n| match n {
                OVERFLOW => 0,
                _ => n,
            })
            .collect()
    }

    async fn execute_gpu(&self, numbers: &[u8]) -> Option<Vec<u8>> {
        // Instantiates instance of WebGPU
        let instance = wgpu::Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        let info = adapter.get_info();
        // skip this on LavaPipe temporarily
        if info.vendor == 0x10005 {
            return None;
        }

        self.execute_gpu_inner(&device, &queue, numbers).await
    }

    async fn execute_gpu_inner(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        numbers: &[u8],
    ) -> Option<Vec<u8>> {
        // Loads the shader from WGSL
        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        // Gets the size in bytes of the buffer.
        let slice_size = numbers.len() * std::mem::size_of::<u8>();
        let size = slice_size as wgpu::BufferAddress;

        // Instantiates buffer without data.
        // `usage` of buffer specifies how it can be used:
        //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
        //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Instantiates buffer with data (`numbers`).
        // Usage allowing the buffer to be:
        //   A storage buffer (can be bound within a bind group and thus available to a shader).
        //   The destination of a copy.
        //   The source of a copy.
        let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer"),
            contents: bytemuck::cast_slice(numbers),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // A bind group defines how buffers are accessed by shaders.
        // It is to WebGPU what a descriptor set is to Vulkan.
        // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

        // A pipeline specifies the operation of a shader

        // Instantiates the pipeline.
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &cs_module,
            entry_point: "main",
        });

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer.as_entire_binding(),
            }],
        });

        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("compute collatz iterations");
            cpass.dispatch_workgroups(numbers.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
        }
        // Sets adds copy operation to command encoder.
        // Will copy data from storage buffer on GPU to staging buffer on CPU.
        encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);

        // Submits command encoder for processing
        queue.submit(Some(encoder.finish()));

        // Note that we're not calling `.await` here.
        let buffer_slice = staging_buffer.slice(..);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        device.poll(wgpu::Maintain::Wait);

        // Awaits until `buffer_future` can be read from
        if let Some(Ok(())) = receiver.receive().await {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result = bytemuck::cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            staging_buffer.unmap(); // Unmaps buffer from memory
                                    // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                    //   delete myPointer;
                                    //   myPointer = NULL;
                                    // It effectively frees the memory

            // Returns data from buffer
            Some(result)
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}

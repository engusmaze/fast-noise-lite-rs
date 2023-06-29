use fast_noise_lite_rs::FastNoiseLite;
use image::{ImageBuffer, Luma};

fn main() {
    let noise = FastNoiseLite::new(1337);

    ImageBuffer::from_fn(1024, 1024, |x, y| {
        let x = x as f32 - 512.0;
        let y = y as f32 - 512.0;
        Luma([(noise.get_noise_2d(x, y) * 128.0 + 128.0) as u8])
    })
    .save("examples/simplex.png")
    .expect("Failed to save image")
}

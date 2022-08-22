use fast_noise_lite_rs::FastNoiseLite;

fn generate_image(mut f: impl FnMut(f32, f32) -> f32, image_name: &str) {
    let mut image: image::RgbImage = image::ImageBuffer::new(1024, 1024);
    for (x, y, rgb) in image.enumerate_pixels_mut() {
        let v = f(x as f32 - 512.0, y as f32 - 512.0);
        let v = (v * 128.0 + 128.0) as u8;
        rgb.0 = [v, v, v];
    }
    image
        .save("examples/".to_owned() + image_name + ".png")
        .unwrap();
}

fn main() {
    let noise = FastNoiseLite::new(1337);

    generate_image(
        |x, y| {
            let x = x;
            let y = y;
            noise.get_noise_2d(x, y)
        },
        "example",
    );
}

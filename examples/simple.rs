use fast_noise_lite_rs::{FastNoiseLite, FractalType, NoiseType};

fn main() {
    let mut noise = FastNoiseLite::new(1337);

    noise.set_noise_type(NoiseType::OpenSimplex2);
    noise.set_frequency(0.02);
    noise.set_fractal_type(FractalType::DomainWarpProgressive);

    let x = 2.5;
    let y = 6.2;
    let noise_value = noise.get_noise_2d(x, y);

    println!("{noise_value:?}");
}

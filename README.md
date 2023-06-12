# hash_art

This is a small tool that tries to find a version of a source image that resembles a target image when individual image blocks are hashed. This is done by slightly modifying the source image and iteratively calculating hashes of the distorted source image. The version . Due to the pseudo random behaviour of hash algorithms this is done in a brute force fashion.

Currently the calculations are done on the CPU but my goal is to have an implementation that uses WebGPU.

## How to use it

### Basic usage

```
hash_art --source source_image.jpg --target target_image.jpg
```

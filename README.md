# Fluid simulation in Rust
## Goal:

- [✔] Solving SPH
- [✔] Surface render
- [❌] Object colisions
- [✔] Exporting results
- [✔] Live view of simulation
- [✔] Fully multithreaded

## Usage:

Compile it with Rust and then you can run the simulation directly. Using `./sph true` will also generate luxrender file in the `render` folder.

To generate the image sequence use the `generate_image.py`. Don't forget to add the Luxrender folder into PYTHONPATH so luxerender can be used inside python.

It take few second to simulate, few hours to generate files and few hours to generate images. 

Then, FFMPEG can be used to generate mp4. 

## Example


https://user-images.githubusercontent.com/13602291/224392462-52a1ca24-8805-4e52-80b6-45b2391d6514.mp4


https://user-images.githubusercontent.com/13602291/224392481-d5ef976f-3431-455f-b722-b991786e548a.mp4


## Credits:

cK0nrad

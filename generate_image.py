import random
import time
import sys
import os
import shutil
from array import *
import pyluxcore

def main():
	pyluxcore.Init()
	pyluxcore.SetLogHandler(None)
	start_time_global = time.time()
	for i in range(0,500):
		input_file = "./render/modele.lxs"
		output_file = "./render/render.lxs"
		shutil.copy(input_file, output_file)

		with open(output_file, 'r') as f:
			contents = f.read()

		replacement_char = str(i)
		modified_contents = contents.replace("$", replacement_char)

		with open(output_file, 'w') as f:
			f.write(modified_contents)

		scnProps = pyluxcore.Properties()
		cfgProps = pyluxcore.Properties()
		pyluxcore.ParseLXS("./render/render.lxs", scnProps, cfgProps)
		scnProps.SetFromString(f"""
			scene.volumes.vol_air.type = homogeneous
			scene.volumes.vol_air.absorption = 0.07 0.07 0.07
			scene.volumes.vol_air.scattering = 0.1 0.1 0.1 
			scene.volumes.vol_air.asymmetry = 0.0 0.0 0.0
			scene.volumes.vol_air.multiscattering = 1
			
			scene.camera.lookat.orig = 50 100 -150
			scene.camera.lookat.target = 50 80 -110
			scene.camera.up = 0 1 0
			scene.camera.fieldofview = 45
			scene.camera.lensradius = 0
			scene.camera.focaldistance = 1e+30
			scene.camera.hither = 0.001
			scene.camera.yon = 1e+30

			scene.lights.skylight.type = sky2
			scene.lights.skylight.dir = 0 0 0.783085
			scene.lights.skylight.turbidity = 3.5
			scene.lights.skylight.gain = .00025 .00025 .00025

			scene.materials.LUXCORE_MATERIAL_LUXCORE_OBJECT_0.type = "matte"
			scene.objects.LUXCORE_OBJECT_0.material = "LUXCORE_MATERIAL_LUXCORE_OBJECT_0"
			scene.objects.LUXCORE_OBJECT_0.vertices = -350 0 -350 350 0 -350 350 0 350 -350 0 350
			scene.objects.LUXCORE_OBJECT_0.faces = 0 1 2 2 3 0
			scene.objects.LUXCORE_OBJECT_0.transformation = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
			scene.objects.LUXCORE_OBJECT_0.uvs = 0 0 5 0 5 5 0 5

			scene.textures.constante_1.type = "constfloat1"
			scene.textures.constante_1.value = 0.7
			
			scene.materials.glass.type = "glass"
			scene.materials.glass.interiorior = "constante_1"
			scene.materials.glass.id = 3364224
			scene.materials.glass.emission.gain = 1 1 1
			scene.materials.glass.emission.power = 0
			scene.materials.glass.emission.efficency = 0
			scene.materials.glass.emission.theta = 90
			scene.materials.glass.emission.samples = -1
			scene.materials.glass.emission.id = 0
			scene.materials.glass.visibility.indirect.diffuse.enable = 1
			scene.materials.glass.visibility.indirect.glossy.enable = 1
			scene.materials.glass.visibility.indirect.specular.enable = 1
			scene.materials.glass.shadowcatcher.enable = 0
			scene.materials.glass.shadowcatcher.onlyinfinitelights = 0
			scene.materials.glass.volume.exterior = vol_air
			
			scene.objects.LUXCORE_OBJECT_5.material = "glass"
			{cfgProps.Get("scene.objects.LUXCORE_OBJECT_5.vertices")}
			{cfgProps.Get("scene.objects.LUXCORE_OBJECT_5.faces")}
			scene.objects.LUXCORE_OBJECT_5.transformation = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
		""")

		cfgProps.SetFromString("""
			opencl.platform.index = -1
			opencl.cpu.use = 0
			opencl.gpu.use = 1
			renderengine.type = "PATHOCL"
			sampler.type = "METROPOLIS"
			film.filter.type = "MITCHELL_SS"
			lightstrategy.type = "UNIFORM"
			accelerator.instances.enable = 0
			film.width = 1920
			film.height = 1080
			film.imagepipeline.0.type = "TONEMAP_AUTOLINEAR"

			film.imagepipeline.1.type = "GAMMA_CORRECTION"
			film.imagepipeline.1.value = 2.0

			film.imagepipeline.2.type = "OPTIX_DENOISER"

			batch.halttime = 150
			batch.haltspp = 0	
			film.outputs.2.filename = normal.png
		""")
		
		scene = pyluxcore.Scene()
		scene.Parse(scnProps)

		config = pyluxcore.RenderConfig(cfgProps, scene)
		session = pyluxcore.RenderSession(config)

		session.Start()

		startTime = time.time()
		while True:
			time.sleep(0.5)
			session.UpdateStats()
			if time.time() - startTime >= 5:
				break

		session.Stop()
		session.GetFilm().Save()
		shutil.move("normal.png", f"./images/water_{str(i).zfill(4)}.png")
		endTime = time.time()
		print(f"{i+1}/{500} Done in {endTime-startTime} | Total : {endTime - start_time_global}.", end="\r")
	
	print(f"\nDone in {time.time() - start_time_global} seconds.")

if __name__ == '__main__':
	main()

import os, sys, shutil, subprocess

libs = ["PLoM_C_library.o","PLoM_C_library.so"]

for cur_lib in libs:
    if os.path.exists(cur_lib):
        os.remove(cur_lib)

if sys.platform == 'win32':
    subprocess.run("make.bat")
    dst_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"lib/win")
else:
    subprocess.run(["make", "all"])
    if sys.platform == 'darwin':
        dst_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"lib/macOS")
    else:
        dst_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"lib/linux")

for cur_lib in libs:
    shutil.copy2(cur_lib, dst_dir)
    os.remove(cur_lib)
    
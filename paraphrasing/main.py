import os

def check_env():
    cwd = os.getcwd()
    bat_file = os.path.join(cwd, ".nlg_env/Scripts/activate.bat")

    # Checking if environment exists or not
    if os.path.exists(bat_file):
        print("\nEnvironment Already Exists...")
        print("Activating Environment...")
        os.system(f'cmd /c {bat_file}')
    else:
        install_activate_env()


def install_activate_env():
    cwd = os.getcwd()
    env_path = os.path.join(cwd, ".nlg_env")
    bat_file = os.path.join(cwd, ".nlg_env/Scripts/activate.bat")
    req_file = os.path.join(cwd, "requirement.txt")

    env_command = f"python -m venv {env_path}"
    req_command = f"pip install -r {req_file}"

    print("\nEnvironment does not exists...")
    print("Creating Environment...")
    os.system(env_command)
    print("Activating Environment ...")
    os.system(f'cmd /c {bat_file}')
    print("Installing dependencies ...")
    os.system(f'cmd /c {req_command}')


if __name__ == "__main__":
    
    process = input("Choose option finetune(f) or inference(i) : ")

    if process == 'f':
        data_dir = input("\nPath to data directory : ")
        trained_model = input("Path to previously finetuned model (if any)/(None): ")
        save_model_path = input("Path to save model: ")

        check_env()

        os.system(f"python finetune.py {data_dir} {trained_model} {save_model_path}")
    
    elif process == 'i':
        model_path = input("\nPath to finetuned model: ")
        inp_file = input("Path to test file: ")
        save_file = input("Path to save result: ")

        check_env()
        
        os.system(f"python inference.py {model_path} {inp_file} {save_file}")

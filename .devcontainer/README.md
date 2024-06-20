# Learn OpenCV — Dev Container Usage

VS Code [Dev Containers](https://code.visualstudio.com/docs/remote/containers) let you run the Learn OpenCV samples inside an isolated development environment powered by [Docker](http://docker.com/) containers. This has some key benefits: 

- **Simple setup** — Just open this repository as a VS Code Dev Container and—after the container builds—you will be working in an isolated environment that already has OpenCV and Python with OpenCV bindings installed.
- **Isolated** — Dev containers are isolated from your local environment so that tools installed inside the container do not pollute your local machine.
- **Full development environment** — You can seamlessly use all the features of VS Code, including the extensions that provide IntelliSense for Python and C++, inside of dev containers. 

## Usage

To run these samples using a dev container:

1. [Setup VS Code Remote Development](https://code.visualstudio.com/docs/remote/containers).
1. Clone this repository to your local machine and open it in VS Code.
1. Run the `Remote — Container: Reopen Folder in Container` command to start the dev Container.
1. Once the container builds, VS Code will launch inside the new, isolated development environment 

If you are working with the Python samples, select the Python environment you wish to use by running `workon` inside the dev container. For example, `workon OpenCV-3.4.3-py3` to use Python3 with OpenCV 3.4.3 bindings.

```bash
workon OpenCV-3.4.3-py3
cd Threshold/
python threshold.py
```

### For GUI support on MacOS

Many samples use UI to display their results (using the `imgShow` function for example). For these samples, you need to configure X11 forwarding for the dev container. To do this on macOS:

1. Install [XQuartz](https://www.xquartz.org)
1. Open XQuartz and go to `XQuartz` -> `Preferences` -> `Security`. Enable connections for network clients.
1. On your local machine, allow connections from your ip by running:

    ```bash
    xhost + $(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
    ```

1. Now open the Lean OpenCV dev container in VS Code.
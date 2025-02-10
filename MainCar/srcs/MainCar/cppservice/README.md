# C++ 17 Blueprint

A pure C++ 17 blueprint with [CMake](https://cmake.org/) build system support. [Poco](https://pocoproject.org/) is used for network connections. The generated code can be enhanced with support for the [Conan](https://conan.io/) package manager.

## Features
The following list presents a set of features which can be individually enabled in the solution file for code generation.

* **api**<br/>
    Create the plain interface data structure files
* **stubs**<br/>
    Provides a basic example implementation, without business logic in operations.
* **scaffold**<br/>
    Adds a structure to project. Adds cmake files for the api and stubs, sets up tests with catch2.
* **monitor**<br/>
    Create support libraries for monitoring API traffic
* **olink**<br/>
    Create IPC [ObjectLink](https://objectlinkprotocol.net/) adapters for your interface classes.
* **mqtt**<br/>
    Create IPC implementation for [MQTT v5](https://mqtt.org/). Provides ready to use client and server for your Interfaces
    Please note this feature is still experimental.
* **examples**<br/>
    Create simple examples for "How to use" the generated code for stubs
* **example-olink**<br/>
    Creates simple example for "How to use" the generated code with olink
* **conan**<br/>
    Create files for conan package manager

* **core**<br/>
    Create a core set of files common for few features. Generated anytime one of features is enabled: stubs, monitor, olink.
* **apigear**<br/>
    Create necessary ApiGear support library for extended features like monitoring, olink IPC - needs to be generated for monitor or olink

## Folder structure
The following gives a simplified overview of the generated code structure with `all` features enabled.

In general, the files in the `modules/modulename1/generated/..` are completely auto generated and should not be modified manually.
If changes to these files are necessary those should be applied in the technology template.

The files in the `modules/modulename1/implementation/..` are meant to be a starting point for the implementation of real business logic for the interface. Those files will not be automatically overwritten by the generators **scaffold** feature unless `force: true` is specified in the solution file.

It is recommended to have at least manually modified files, like the `implementation` folder, under source code version control. This can also be helpful to spot differences in the auto generated code, e.g. after API changes or a new version of the template, generator.

```
outputfolder
|   README.md
|   CMakeLists.txt
│   test_conan.sh
│   test_conan.bat
│
└───apigear
│   │   ...
│
└───examples
│   │   ...
│   
└───modules
    └───modulename1_module
        └─── conanfile.py
        └─── test_package
		└─── modulename1
				|  CMakeLists.txt
				|   modulename.cmake.in
				└───generated
				|   └───api
				|   |    | ...
				|   |
				|   └───core
				|   |    | ...
				|   |
				|   └───monitor
				|   |    | ...
				|   |
				|   └───olink
				|        | ...
				└───implementation
```
## Build for development

The template supports two build methods. Depending on your environment you can choose between a [pure CMake](#CMake-pure) approach and additional one with [Conan support](#Conan-support).

### CMake pure

If you want to manage and install dependencies manually into the environment, the plain CMakeLists.txt approach will just work fine. 
[Documentation for CMake](https://cmake.org/)

#### Windows Setup

For ease of use, we use the Conan package manager to install needed dependencies first. [Documentation for Conan](https://conan.io/).
If you do not want or can not use conan, the poco libraries must installed separately.

1. If you have not done before, install conan 1.x and then set up your profile. The default configuration is usually in your home folder at `.conan/profiles/default`. More details on the conan profile can be found [here](https://docs.conan.io/en/latest/reference/profiles.html).
   This can be set like:
    ```
    PS C:\project_source> conan profile detect
    PS C:\project_source> conan profile update settings.compiler.runtime=MDd default
    PS C:\project_source> conan profile update settings.build_type=Debug default
    ```
2. Use conan to install the Poco libraries
   This step pulls the poco library from the conan-center index and builds it if necessary. Afterwards, the needed CMake files are generated in this folder for other projects to use the library.
   We create those CMake files in a dedicated `deps` folder which we need to specify in the next step.
    ```
    PS C:\project_source> mkdir deps
    PS C:\project_source> cd deps
    PS C:\project_source\deps> conan install poco/1.11.3@ --build missing -o poco/*:shared=False -o poco/*:enable_data_mysql=False -o poco/*:enable_activerecord=False -o poco/*:enable_apacheconnector=False -o poco/*:enable_cppparser=False -o poco/*:enable_crypto=True -o poco/*:enable_data=False -o poco/*:enable_data_odbc=False -o poco/*:enable_data_postgresql=False -o poco/*:enable_data_sqlite=False -o poco/*:enable_encodings=False -o poco/*:enable_json=False -o poco/*:enable_jwt=False -o poco/*:enable_mongodb=False -o poco/*:enable_net=True -o poco/*:enable_netssl=True -o poco/*:enable_pagecompiler=False -o poco/*:enable_pagecompiler_file2page=False -o poco/*:enable_pdf=False -o poco/*:enable_pocodoc=False -o poco/*:enable_redis=False -o poco/*:enable_sevenzip=False -o poco/*:enable_util=True -o poco/*:enable_xml=False -o poco/*:enable_zip=False --generator cmake_find_package --generator virtualenv
    PS C:\project_source\deps> cd ..
    ```
3. Build the project
   First the project sources need to be configured and in a second step they are built. The project will be built in the folder `build`.
   For CMake to find the dependencies we need to point the `CMAKE_MODULE_PATH` variable to the path used in the previous step.
   If the files shall be re-used, they can be installed in a location specified by `CMAKE_INSTALL_PREFIX`
   ```
   PS C:\project_source> cmake -Bbuild -DCMAKE_INSTALL_PREFIX=tmp -DCMAKE_PREFIX_PATH=C:/project_source/deps
   PS C:\project_source> cmake --build build/
   ```
4. Install binaries
   The last step installs the previously built executables, libraries and headers files in the folder specified `CMAKE_INSTALL_PREFIX` above.
   ```
   PS C:\project_source> cmake --build build/ --target install
   ```
#### Ubuntu 20.04 Setup

1. Install needed packages

    ```
    $ sudo apt install gcc-10 g++-10 git cmake libpoco-dev
    ```
2. Set gcc-10 with C++17 support as default compiler
   
    ```
    $ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10 --slave /usr/bin/g++ g++ /usr/bin/g++-10
    $ sudo update-alternatives --config gcc
    ```

3. Ubuntu 20.04 does not provide a newer nlohmann::json version within the packages sources. Therefore it has to be manually downloaded and installed.

    ```
    $ git clone --branch v3.11.3 --depth 1 https://github.com/nlohmann/json.git
    $ cd json
    $ cmake -Bbuild -H. -DBUILD_TESTING=OFF
    $ sudo cmake --build build/ --target install
    ```

4. Ubuntu 20.04 does not provide Catch2 within the packages sources. Therefore it has to be manually downloaded and installed.

    ```
    $ git clone --branch v2.13.8 --depth 1 https://github.com/catchorg/Catch2.git
    $ cd Catch2
    $ cmake -Bbuild -H. -DBUILD_TESTING=OFF
    $ sudo cmake --build build/ --target install
    ```

5. Build the generated code

    ```
    $ cmake -Bbuild -H.
    $ sudo cmake --build build/ --target install
    ```


#### Minimum Poco build

In case you want to build a minimum poco version. You need to install the necessary dependencies like described [here](https://github.com/pocoproject/poco#prerequisites) and then do a minimal configuration.

```
$ mkdir cmake-build && cd cmake-build
$ cmake -DENABLE_ACTIVERECORD=OFF -DENABLE_ACTIVERECORD_COMPILER=OFF -DENABLE_APACHECONNECTOR=OFF -DENABLE_CPPPARSER=OFF -DENABLE_CRYPTO=ON -DENABLE_DATA=OFF -DENABLE_DATA_MYSQL=OFF -DENABLE_DATA_ODBC=OFF -DENABLE_DATA_POSTGRESQL=OFF -DENABLE_DATA_SQLITE=OFF -DENABLE_ENCODINGS=OFF -DENABLE_ENCODINGS_COMPILER=OFF -DENABLE_FOUNDATION=ON -DENABLE_JSON=OFF -DENABLE_JWT=OFF -DENABLE_MONGODB=OFF -DENABLE_NET=ON -DENABLE_NETSSL=ON -DENABLE_PAGECOMPILER=OFF -DENABLE_PAGECOMPILER_FILE2PAGE=OFF -DENABLE_PDF=OFF -DENABLE_POCODOC=OFF -DENABLE_REDIS=OFF -DENABLE_SEVENZIP=OFF -DENABLE_TESTS=OFF -DENABLE_UTIL=ON -DENABLE_XML=OFF -DENABLE_ZIP=OFF ..
$ make -j
```

### Conan support

For ease of use and package distribution we generate all files necessary files to use the Conan package manager.
[Documentation for Conan](https://conan.io/)
1. Set your conan profile to use C++11 and newer. The default configuration is usually at `.conan/profiles/default`. More details on the conan profile can be found [here](https://docs.conan.io/en/latest/reference/profiles.html).
   This can be set like:

    ```
    conan profile update settings.compiler.libcxx=libstdc++11 default
    ```
2. Execute the test script depending on the host platform. This will build all dependencies and module files, including examples.
   * Or on **Linux, Mac** execute the shell script `scripts/test_conan.sh`.

        ```
        $ bash scripts/test_conan.sh
        ```
    * On **Windows** execute the batch script `scripts/test_conan.bat`.

        ```
        >scripts/test_conan.bat
        ```

## Build for distribution

In some cases you may want to share the build artifacts. The easiest approach is to build everything in release mode and link statically against external libraries.
In general the following steps are similar to the development build. We use conan to build the static version of the poco libraries.

The procedure is almost identical for Windows and Linux, but it is important to use the right path format corresponding to each platform.

1. If you have not done before, install conan 1.x and then set up your release profile. More details on the conan profile can be found [here](https://docs.conan.io/en/latest/reference/profiles.html).
   
    A new profile can be created like this:
    ```
    $ conan profile detect --name release
    ```

    And additionally we need to make platform specific changes to the conan configuration:
    * On **Linux**:

        ```
        $ conan profile update settings.compiler.libcxx=libstdc++11 release
        $ conan profile update settings.build_type=Release release
        ```
    * On **Windows** 

        ```
        > conan profile update settings.compiler.runtime=MD release
        > conan profile update settings.build_type=Release release
        ```

2. Once conan is set up, we use it to install the necessary dependencies and build them if not available.
    It is recommended to do this in a new folder, which we later use to configure cmake.
    For instance, create a folder ` deps` and simply copy the `conan install` line:
    ```
    $ mkdir deps
    $ cd deps
    $ conan install --profile=release poco/1.11.3@ --build missing -o poco/*:shared=False -o poco/*:enable_data_mysql=False -o poco/*:enable_activerecord=False -o poco/*:enable_apacheconnector=False -o poco/*:enable_cppparser=False -o poco/*:enable_crypto=True -o poco/*:enable_data=False -o poco/*:enable_data_odbc=False -o poco/*:enable_data_postgresql=False -o poco/*:enable_data_sqlite=False -o poco/*:enable_encodings=False -o poco/*:enable_json=False -o poco/*:enable_jwt=False -o poco/*:enable_mongodb=False -o poco/*:enable_net=True -o poco/*:enable_netssl=True -o poco/*:enable_pagecompiler=False -o poco/*:enable_pagecompiler_file2page=False -o poco/*:enable_pdf=False -o poco/*:enable_pocodoc=False -o poco/*:enable_redis=False -o poco/*:enable_sevenzip=False -o poco/*:enable_util=True -o poco/*:enable_xml=False -o poco/*:enable_zip=False --generator cmake_find_package --generator virtualenv
    $ cd ..
    ```
    The `conan install` step downloads the poco source package, configures it to a minium version and builds it, including its dependencies.
    Also make sure to use the previously created release profile with `--profile=release`.
3. With poco available in the `deps` folder we can configure our project and build it.

    The `CMAKE_INSTALL_PREFIX` defines the where built files shall be installed. In this example we use `tmp`.
    The `CMAKE_MODULE_PATH` tells CMake where it can find the necessary dependencies like poco. In the previous step we used `deps` - so we specify the full path to the deps folder here, e.g. "/home/user/project/deps" or "C:/workspace/project/deps".
    ```
    $ cmake -Bbuild -DCMAKE_INSTALL_PREFIX=tmp -DCMAKE_PREFIX_PATH="/home/user/project/deps" -DCMAKE_BUILD_TYPE=Release
    ```

    After the configuration step finished successfully we can build the project
    ```
    $ cmake --build build/ --config Release
    ```
4. The last step is to install the built project files into our previously specified folder("tmp") and package it.

    ```
    $ cmake --build build/ --target install --config Release
    ```

    The files are in "tmp/bin", "tmp/lib" and "tmp/include".

5. In order to run the files on Windows it is sufficient to just execute the following command, e.g.:
    ```
    > tmp\bin\OLinkServer.exe
    ```

    For Linux, it is necessary to point to the dynamically linked executables since they are in the "lib" folder.
    ```
    $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/project/tmp/lib
    $ ./tmp/bin/OLinkServer
    ```

## Run
Once the build was successful you can easily launch one of the examples applications.

### CMake
- Execute one of the examples on Linux or Mac

    ```
    $ ./build/bin/OLinkServer
    ```
- Execute one of the examples on Windows

    ```
    $ PS C:\project_source> .\build\bin\Debug\OLinkServer.exe
    ```
### Conan

* On **Linux, Mac** follow these steps
  1. Source the conan [virtual environment](https://docs.conan.io/en/latest/mastering/virtualenv.html) with all built dependency libraries.

        ```
        $ source ./build/examples/olinkserver/activate.sh
        (conanenv)
        ```
  2. Execute one of the examples

        ```
        $ ./build/examples/olinkserver/OLinkServer.exe
        [info   ] RemoteRegistry.addObjectSource: modulename.Interfacename
        wss.listen() on 8000
        Enter command:
        ```
* On **Windows** follow these steps
  1. Source the conan [virtual environment](https://docs.conan.io/en/latest/mastering/virtualenv.html) with all built dependency libraries.

        ```
        >build\examples\olinkserver\activate.bat
        (conanenv)
        ```
  2. Execute one of the examples

        ```
        >build\examples\olinkserver\Debug\OLinkServer.exe
        [info   ] RemoteRegistry.addObjectSource: modulename.Interfacename
        wss.listen() on 8000
        Enter command:
        ```






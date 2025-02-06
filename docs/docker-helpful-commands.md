# Docker Helful Commands

### 1. **WORKDIR**
   - **Description**: Sets the working directory inside the container. If the directory doesn’t exist, it will be created.
   - **Example**:
     ```dockerfile
     WORKDIR /app
     ```
     This will set `/app` as the working directory for any subsequent instructions like `RUN`, `CMD`, etc.

### 2. **EXPOSE**
   - **Description**: Tells Docker that the container listens on the specified network ports at runtime. This does not actually publish the port; it’s just for documentation purposes and can be used in combination with the `-p` option when running the container.
   - **Example**:
     ```dockerfile
     EXPOSE 80
     ```
     This exposes port 80 inside the container.

### 3. **VOLUME**
   - **Description**: Creates a mount point with the specified path and marks it as holding externally mounted volumes. It allows data to persist even if the container is removed.
   - **Example**:
     ```dockerfile
     VOLUME /data
     ```
     This will create a `/data` directory in the container for external volume mounting.

### 4. **ENV**
   - **Description**: Sets an environment variable in the container.
   - **Example**:
     ```dockerfile
     ENV NODE_ENV production
     ```
     This sets the `NODE_ENV` variable to `production`.

### 5. **COPY**
   - **Description**: Copies new files or directories from the host machine into the container’s filesystem.
   - **Example**:
     ```dockerfile
     COPY ./app /usr/src/app
     ```
     This will copy the contents of the `./app` directory into `/usr/src/app` in the container.

### 6. **ADD**
   - **Description**: Similar to `COPY`, but with more features, such as automatically extracting tar files and downloading files from a URL.
   - **Example**:
     ```dockerfile
     ADD myapp.tar.gz /usr/src/app
     ```
     This will add and extract the `myapp.tar.gz` archive into `/usr/src/app`.

### 7. **RUN**
   - **Description**: Executes a command in the container. Typically used for installing software and dependencies.
   - **Example**:
     ```dockerfile
     RUN apt-get update && apt-get install -y curl
     ```
     This will update the package lists and install `curl` in the container.

### 8. **CMD**
   - **Description**: Specifies the command to run when the container starts. It can be overridden by providing a command when running the container.
   - **Example**:
     ```dockerfile
     CMD ["node", "server.js"]
     ```
     This will run `node server.js` when the container is started.

### 9. **ENTRYPOINT**
   - **Description**: Defines a command that cannot be overridden when starting the container. It’s similar to `CMD`, but the provided command in `CMD` will be appended to the `ENTRYPOINT` command.
   - **Example**:
     ```dockerfile
     ENTRYPOINT ["python3"]
     CMD ["app.py"]
     ```
     This will run `python3 app.py` when the container starts.

### 10. **ARG**
   - **Description**: Defines a build-time argument that can be passed to the `docker build` command.
   - **Example**:
     ```dockerfile
     ARG VERSION=1.0
     ```
     This allows you to specify a version value when building the Docker image, like `docker build --build-arg VERSION=2.0 .`.

Sure! Below are additional Docker commands that you may find useful to include in a separate documentation file.

### 11. **USER**
   - **Description**: Sets the user name or UID and optionally the group name or GID to use when running the container.
   - **Example**:
     ```dockerfile
     USER node
     ```
     This sets the user `node` to run the container.

### 12. **LABEL**
   - **Description**: Adds metadata to the image, which can be used for organizational purposes, automation, or querying.
   - **Example**:
     ```dockerfile
     LABEL version="1.0" description="This is a Node.js app"
     ```
     This adds the `version` and `description` metadata to the image.

### 13. **HEALTHCHECK**
   - **Description**: Specifies a command to check the health of the container. This is used to monitor if the container is still working correctly.
   - **Example**:
     ```dockerfile
     HEALTHCHECK CMD curl --fail http://localhost:8080/health || exit 1
     ```
     This checks the health of the container by trying to access a health check URL.

### 14. **STOPSIGNAL**
   - **Description**: Specifies the system call signal that will be sent to the container to stop it. This is useful for controlling the way a container is shut down.
   - **Example**:
     ```dockerfile
     STOPSIGNAL SIGTERM
     ```
     This sends `SIGTERM` to stop the container gracefully.

### 15. **SHELL**
   - **Description**: Defines the shell to use for the `RUN` command.
   - **Example**:
     ```dockerfile
     SHELL ["/bin/bash", "-c"]
     ```
     This sets `/bin/bash -c` as the shell used for `RUN` instructions.

### 16. **FROM**
   - **Description**: Specifies the base image to use for the container. This is the first instruction in a Dockerfile.
   - **Example**:
     ```dockerfile
     FROM node:14
     ```
     This specifies that the container should be based on the `node:14` image.

### 17. **ARG**
   - **Description**: Defines build-time variables that can be passed to `docker build`.
   - **Example**:
     ```dockerfile
     ARG app_version=1.0
     ```
     This allows passing the `app_version` as a build argument.

### 18. **docker build**
   - **Description**: Builds an image from a Dockerfile.
   - **Example**:
     ```bash
     docker build -t my-app .
     ```
     This builds an image from the current directory’s `Dockerfile` and tags it as `my-app`.

### 19. **docker run**
   - **Description**: Runs a container from a specified image.
   - **Example**:
     ```bash
     docker run -d -p 8080:80 --name my-container my-app
     ```
     This runs a container from the `my-app` image, mapping port 8080 on the host to port 80 in the container.

### 20. **docker ps**
   - **Description**: Lists all running containers.
   - **Example**:
     ```bash
     docker ps
     ```
     This command shows a list of currently running containers.

### 21. **docker stop**
   - **Description**: Stops a running container.
   - **Example**:
     ```bash
     docker stop my-container
     ```
     This stops the container named `my-container`.

### 22. **docker exec**
   - **Description**: Runs a command in a running container.
   - **Example**:
     ```bash
     docker exec -it my-container bash
     ```
     This opens an interactive bash shell inside the running `my-container` container.

### 23. **docker logs**
   - **Description**: Fetches logs from a container.
   - **Example**:
     ```bash
     docker logs my-container
     ```
     This shows the logs from the `my-container` container.

### 24. **docker images**
   - **Description**: Lists all images on the local machine.
   - **Example**:
     ```bash
     docker images
     ```
     This lists all images available locally.

### 25. **docker rmi**
   - **Description**: Removes one or more images.
   - **Example**:
     ```bash
     docker rmi my-app
     ```
     This removes the `my-app` image from the local machine.

### 26. **docker volume**
   - **Description**: Manages Docker volumes, which store persistent data.
   - **Example**:
     ```bash
     docker volume create my-volume
     ```
     This creates a new volume called `my-volume`.

### 27. **docker network**
   - **Description**: Manages Docker networks, which allow containers to communicate.
   - **Example**:
     ```bash
     docker network create my-network
     ```
     This creates a new Docker network called `my-network`.

### 28. **docker-compose**
   - **Description**: A tool to define and manage multi-container Docker applications using a `docker-compose.yml` file.
   - **Example**:
     ```bash
     docker-compose up
     ```
     This starts up the services defined in the `docker-compose.yml` file.

---
# Docker Starter Guide

## 📌 Introduction
This guide provides essential Docker commands for managing **images, containers, volumes, networks, and logs**. Use this as a reference for setting up and working with Docker efficiently.

---

## 🛠️ Docker Installation

Docker is a platform that enables developers to package applications into containers. Below are the steps to install Docker on different operating systems.

### 🔹 Install Docker on macOS via Homebrew
```sh
brew install --cask docker
```
- `--cask` : Installs GUI applications or software requiring additional setup, like Docker Desktop, which includes the Docker daemon.

Alternatively, Docker can be installed without the `--cask` flag:
```sh
brew install docker
brew install colima
colima start
```
- Installs the CLI version of Docker without Docker Desktop.
- `colima` is required to run Docker on macOS when installed without the `--cask` flag.

### 🔹 Install Docker Compose (Optional)
Docker Compose is a tool used to define and manage multi-container applications. If using Colima, Docker Compose must be installed separately:
```sh
brew install docker-compose
```
Docker Compose is optional but useful for running multi-container applications with ease.

### 🔹 Install Docker on Windows (Official Website)
Download and install Docker from the official website:
[Docker Install for Windows](https://docs.docker.com/desktop/install/windows-install/)

After installation, start Docker from the Applications folder or run:
```sh
open -a Docker
```

### 🔹 Check if Docker is installed
Before starting Docker, make sure to start `colima` by running:  
```sh
colima start
```

```sh
docker --version
```

If you face issues starting Docker, quit your terminal and restart it.

### 🔹 Start the Docker service if not running
```sh
docker info
```

---

## 📦 Docker Image Management

Docker images are the blueprint for running containers. These commands help manage images efficiently.

### 🔹 List all images
```sh
docker images
```
- Displays all locally stored Docker images.

### 🔹 Pull an image from Docker Hub
```sh
docker pull nginx:alpine
```
- Downloads the `nginx:alpine` image from Docker Hub.

### 🔹 Build an image from a Dockerfile
```sh
docker build -t my-app .
```
- `-t my-app` : Assigns a name (tag) to the image.
- `.` : Uses the current directory as the build context.

### 🔹 Remove an image
```sh
docker rmi my-app
```
- Deletes the specified Docker image.

---

## 🏗️ Docker Container Management

Containers are instances of images that run applications. These commands help manage containers.

### 🔹 Run a container
```sh
docker run -d -p 80:80 --name my-container nginx
```
- `-d` : Runs the container in detached mode.
- `-p 80:80` : Maps port 80 on the container to port 80 on the host.
- `--name my-container` : Assigns a custom name to the container.

### 🔹 List running containers
```sh
docker ps
```
- Shows only currently running containers.

### 🔹 List all containers (including stopped ones)
```sh
docker ps -a
```
- Displays all containers, whether running or stopped.

### 🔹 Stop a running container
```sh
docker stop my-container
```
- Gracefully stops a running container.

### 🔹 Restart a container
```sh
docker restart my-container
```
- Stops and starts the container again.

### 🔹 Remove a container
```sh
docker rm my-container
```
- Deletes the specified container.

### 🔹 Remove all stopped containers
```sh
docker container prune
```
- Deletes all stopped containers to free up space.

---

## 🔍 Logs & Debugging

Debugging and monitoring container logs help troubleshoot issues efficiently.

### 🔹 View container logs
```sh
docker logs my-container
```
- Displays logs generated by the container.

### 🔹 Enter a running container
```sh
docker exec -it my-container sh
```
- `-it` : Runs in interactive mode.
- `sh` : Opens a shell inside the container.

### 🔹 Check container resource usage
```sh
docker stats
```
- Shows real-time CPU, memory, and network usage of running containers.

---

## 🗂️ Volumes & Cleanup

Docker volumes store persistent data. Proper cleanup helps manage disk space efficiently.

### 🔹 List all volumes
```sh
docker volume ls
```
- Displays all created Docker volumes.

### 🔹 Remove a volume
```sh
docker volume rm my-volume
```
- Deletes the specified volume.

### 🔹 Remove all unused images, containers, and volumes
```sh
docker system prune -a
```
- Frees up disk space by removing unused resources.

---

## 🌐 Docker Network

Docker networks enable communication between containers. By default, Docker creates a `bridge` network, but custom networks can be created for better isolation.

### 🔹 List networks
```sh
docker network ls
```
- Displays all Docker networks.

### 🔹 Create a new network
```sh
docker network create my-network
```
- Creates a custom network.

### 🔹 Run a container in a custom network
```sh
docker run -d --name my-app --network my-network nginx
```
- Connects the container to `my-network`.

---

## 🚀 Running Nginx with Docker

Nginx is a widely used web server. Below are different ways to deploy it with Docker.

### **Basic Nginx Container**
```sh
docker run -d -p 80:80 --name nginx-server nginx:alpine
```
- Runs an Nginx container on port 80.

### **Using a Custom `nginx.conf`**
```sh
# Create a custom config file
nano nginx.conf
```
Example `nginx.conf`:
```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;
}
```

### **Modify Dockerfile to Use Custom Nginx Config**
```dockerfile
FROM nginx:alpine
COPY dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### **Build and Run**
```sh
docker build -t my-nginx-app .
docker run -d -p 80:80 --name my-nginx my-nginx-app
```

Access the app at **http://localhost**

---

## 🔥 Cleanup and Stopping Docker

Proper cleanup ensures efficient resource management.

### 🔹 Stop all running containers
```sh
docker stop $(docker ps -q)
```
- Stops all currently running containers.

### 🔹 Remove all containers
```sh
docker rm $(docker ps -aq)
```
- Deletes all containers, including stopped ones.

### 🔹 Remove all images
```sh
docker rmi $(docker images -q)
```
- Deletes all local images.

---

## See Other Useful Docker Commands

In addition to the basic Docker commands listed in this guide, there are many other useful commands that can help streamline your development and deployment processes.

- [Additional Docker commands](docker-helpful-commands.md)
- [Docker compose guide](docker-compose-guide.md)

For more details, visit [Docker Documentation](https://docs.docker.com/).

We will continue updating this guide with more advanced commands and use cases, so stay tuned for further updates!

---

## 🎯 Conclusion
This guide covers the essential Docker commands for managing images, containers, volumes, networks, and logs. Keep this handy as a quick reference! 🚀

# PPCX App

Containerized Django + PosgreSQL (+PostGIS) stack with two main services:
- `db`: PostgreSQL with PostGIS extension for spatial data support.
- `web`: Django application server using Django.

## Setup and Configuration

Clone the repository:
```bash
git clone --recurse-submodules git@github.com:planpincieux/dashboard.git ppcx-dashboard
cd ppcx-dashboard
```

### 1. Set up secrets

This stack uses Docker secrets, mounted as files in containers under /run/secrets. Secrets are not exposed as environment variables.

Required secrets (create once on the host):
- ~/secrets/db_password
- ~/secrets/django_secret_key

Create and secure the secrets:.

```bash
mkdir -p ~/secrets
chmod 700 ~/secrets

printf 'your-db-password\n' > ~/secrets/db_password
printf 'your-django-secret-key\n' > ~/secrets/django_secret_key
chmod 600 ~/secrets/*
```

Notes:
- docker-compose.yaml references these files as secrets; they are mounted read-only into the web and db containers.
- If ~ is not expanded on your system, replace it with the absolute path (/home/fioli/secrets) in docker-compose.yaml.

### 2. Update docker-compose.yaml

Edit `docker-compose.yaml` to ensure the paths to your secrets are correct. Replace `~/secrets/db_password` and `~/secrets/django_secret_key` with the absolute paths if necessary.

Update volume paths if you want to change where data is stored on the host.

PostgreSQL data is persisted in a Docker named volume db_data. Recreating the stack will not delete data unless you explicitly remove the volume: 
  - Volume name (actual name will be prefixed by the Compose project, e.g., ppcx-app_db_data)
  - Mount path in the db container: /var/lib/postgresql/data

### 3. Start the stack

Build and run:
```bash
docker compose up -d
```

Run migrations:
```bash
docker compose exec web python manage.py migrate
```

Access the app:
- http://localhost:8080 (or `http://<your-server-ip>:8080`)

Connect to Postgres from the host (optional):
```bash
psql "postgresql://postgres:$(cat ~/secrets/db_password)@localhost:5434/planpincieux"
```

## Developer quick reference

### Docker useful commands

List of useful commands to work with the docker container:

- Make migration inside docker container: 

```bash
docker compose exec web python manage.py makemigrations ppcx_app
docker compose exec web python manage.py migrate
```

- Connect shell to container:
```bash
docker compose exec web /bin/bash
```

- Install new dependencies and restart the container:

```bash
docker compose exec web python -m pip install <package>
docker compose restart web
```

- Rebuild images without cache:
```bash
docker compose build --no-cache
```

- Run the development server with Django's runserver (with auto-reload on code changes) for development purposes:

```bash
docker compose exec web python manage.py runserver 0.0.0.0:8000
```

### Populate the database with images and DIC

To populate the database with images and DIC data, you can use the scripts provided in the `app` directory.
These scripts can be run inside the docker container using `docker compose exec` or from outside the container if you have the necessary dependencies installed and you have access to the database.

### Rotating secrets

1) Update the file under ~/secrets (e.g., echo 'newpass' > ~/secrets/db_password).
2) Redeploy:
```bash
docker compose up -d
```
The container will read the updated secret from /run/secrets on restart.

## Backup and Restore

### Database

To backup and restore the PostgreSQL database, you can use the following commands:

```bash
export PGPASSWORD="$(cat ~/secrets/db_password)"
pg_dump -h <your-server-ip> -p 5434 -U postgres -d planpincieux -Fc -f /path/to/backups/planpincieux_YYYY-MM-DD.dump
unset PGPASSWORD
```

Restore (restore into an existing database; add -c to clean before restore if desired):
```bash
export PGPASSWORD="$(cat ~/secrets/db_password)"
pg_restore -h <your-server-ip> -p 5434 -U postgres -d planpincieux -v /path/to/backups/planpincieux_YYYY-MM-DD.dump
unset PGPASSWORD
```

Notes:
- The password file used above is ~/secrets/db_password as in this project; adjust path if different.
- Use createdb to create the target database before restore if it does not exist:
```bash
export PGPASSWORD="$(cat ~/secrets/db_password)"
createdb -h <your-server-ip> -p 5434 -U postgres planpincieux
unset PGPASSWORD
```
- The dump format (-Fc) is the custom format produced by pg_dump and restored with pg_restore.


### DIC data directory

#### Backup DIC data

DIC data is stored in a Docker named volume `ppcx-app_dic_data`. 
To back it up, you can use a temporary container to archive the volume content.

```bash
BACKUP_DIR=/home/fioli/storage/francesco/ppcx_db/backups/dic_data
mkdir -p "$BACKUP_DIR"
docker run --rm \
  -v ppcx-app_dic_data:/data \
  -v "$BACKUP_DIR":/backup \
  alpine:3.20 sh -c 'tar czf /backup/dic_data_$(date +%F_%H%M%S).tar.gz -C /data .'
```

This can be automated using cron.

#### Restore DIC data from backup. **IMPORTANT: This will overwrite current volume content**

1. Stop the app
```bash
docker compose stop web
```

2. Restore from backup (adjust BACKUP path)
```bash
BACKUP=/home/fioli/backups/dic_data/dic_data_YYYY-MM-DD_HHMMSS.tar.gz
docker run --rm -v ppcx-app_dic_data:/data alpine:3.20 sh -c 'rm -rf /data/*'
docker run --rm \
  -v ppcx-app_dic_data:/data \
  -v "$(dirname "$BACKUP")":/backup \
  alpine:3.20 sh -c 'tar xzf /backup/'"$(basename "$BACKUP")"' -C /data'

```

3. Sanity check
```bash
docker run --rm -v ppcx-app_dic_data:/data alpine:3.20 sh -c 'ls -lah /data | head'
```

4. Start the app again
```bash

docker compose up -d
```

#### Copy existing DIC data into the ppcx-app_dic_data volume

1. Stop the app to avoid writes during migration
```bash
docker compose stop web
``` 

2. Optional: Ensure the volume exists (created by compose or now)
```bash
docker volume create ppcx-app_dic_data >/dev/null
```

3. Copy from host folder into the volume (preserves metadata)
```bash
SOURCE_DIR=/home/fioli/storage/francesco/ppcx_db/db_data.prev
# Adjust SOURCE_DIR to your actual data location
docker run --rm \
  -v ppcx-app_dic_data:/dest \
  -v "$SOURCE_DIR":/src:ro \
  alpine:3.20 sh -c 'cd /src && tar cf - . | tar xf - -C /dest'
```
  
4. Sanity check
```bash
docker run --rm -v ppcx-app_dic_data:/dest alpine:3.20 sh -c 'ls -lah /dest | head'
```

5. Start the app again
```bash
docker compose up -d
```

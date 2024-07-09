#!/bin/bash

# Support loading secrets from files using the _FILE suffix convention
for _env_file in $(env | grep '_FILE=' | awk -F '=' '{print $1}'); do
  _env_var=$(echo "${_env_file}" | sed -r 's/(.*)_FILE/\1/')
  if [ -f "${!_env_file}" ]; then
    export "${_env_var}"="$(cat "${!_env_file}")"
  fi
done

python -m routellm.openai_server $SERVER_ARGS

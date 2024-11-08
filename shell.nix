{}:
let
  sources = import ./nix/sources.nix;
  nixpkgs = sources.nixpkgs;
  pkgs = import nixpkgs { };
in
(pkgs.buildFHSUserEnv {
  name = "tud-project-dev-shell";
  targetPkgs = pkgs: (with pkgs; [
    python3
    python3Packages.pip
    python3Packages.virtualenv
    stdenv.cc.cc.lib
    pythonManylinuxPackages.manylinux2014Package
  ]);

  profile = ''
    export PS1="(tetris) $PS1"
    export PYTHONDONTWRITEBYTECODE=1
    export HISTFILE="$(realpath .)/.bash_history"
  '';

  runScript = "${pkgs.writeShellScriptBin "runScript" (''
    set -e

    # Create virtualenv if it doesn't exist
    if [ ! -d .venv ]; then
      python3 -m venv .venv
    fi

    # Activate virtualenv and install packages
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

    exec bash
  '')}/bin/runScript";
}).env

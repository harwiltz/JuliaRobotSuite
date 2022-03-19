{
  pkgs ? import <nixpkgs> {},
  nix-ros-overlay ?  import (builtins.fetchTarball {
    url = https://github.com/lopsided98/nix-ros-overlay/archive/master.tar.gz;
  }) {}
}:

with nix-ros-overlay;

mkShell {
  nativeBuildInputs = [ rosPackages.noetic.xacro ];
}

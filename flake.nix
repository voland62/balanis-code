{
  description = "A basic flake with a shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.systems.url = "github:nix-systems/default";
  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
    inputs.systems.follows = "systems";
  };
  inputs.qucs-rflayout = {
    url = "github:thomaslepoix/Qucs-RFlayout";
    inputs.nixpkgs.follows = "nixpkgs";
    inputs.flake-utils.follows = "flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, qucs-rflayout, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
        {
          devShells.default = with pkgs; mkShell {
            packages = [
              bashInteractive
              uv
              nodejs
              (python312.withPackages (ps: with ps; [
                numpy
                matplotlib
                python-openems
              ]))
              openems
              appcsxcad
              qucs-s
              # Native NixOS package; upstream's default uses an impure NixGL wrapper.
              qucs-rflayout.packages.${system}.qucsrflayout
              # typescript
              # python312Packages.pycairo
            ];
            shellHook = lib.optionalString stdenv.hostPlatform.isLinux ''
              # Preserve desktop data paths and derive the GTK schema path from nixpkgs.
              export XDG_DATA_DIRS="${glib.getSchemaDataDirPath gtk3}''${XDG_DATA_DIRS:+:$XDG_DATA_DIRS}"
            '';
          };
          env = nixpkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
            # Python libraries often load native shared objects using dlopen(3).
            # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
            LD_LIBRARY_PATH = nixpkgs.lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
          };

        }
    );
}

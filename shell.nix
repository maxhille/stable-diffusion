let
  nixpkgs = import <nixpkgs> {};
in
  with nixpkgs;
  mkShell rec {
    name = "stable-diffusion-webui";
    buildInputs = [
      libdrm
      stdenv.cc.cc.lib
	  libGL
	  glib
	  zlib
    ];
    shellHook = ''
    export LD_LIBRARY_PATH="${lib.makeLibraryPath buildInputs}";
    '';
  }

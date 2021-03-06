using BinDeps

@BinDeps.setup

enable_sensitivities = true

if enable_sensitivities
    cvodes = library_dependency("libsundials_cvodes")
    idas = library_dependency("libsundials_idas")
else
    cvode = library_dependency("libsundials_cvode")
    ida = library_dependency("libsundials_ida")
end
kinsol = library_dependency("libsundials_kinsol")
nvecserial = library_dependency("libsundials_nvecserial")
sundialslibs = enable_sensitivities ? [cvodes, idas, kinsol, nvecserial] : [cvode, ida, kinsol, nvecserial]

sundialsver = "sundials-2.5.0"
provides(Sources,
    URI("http://ftp.mcs.anl.gov/pub/petsc/externalpackages/$sundialsver.tar.gz"),
    SHA="9935760931fa6539edd0741acbcf4986770426fd5ea40e50ad4ebed0fc77b0d3",
    sundialslibs)

provides(Binaries, URI("https://bintray.com/artifact/download/tkelman/generic/$sundialsver.7z"),
    SHA="81b168ee98a680e9db80a5de88a5fcc5c05ce81eeb0b211f8352996972a026e6",
    sundialslibs, unpacked_dir="usr$(Sys.WORD_SIZE)/bin", os = :Windows)

prefix = joinpath(BinDeps.depsdir(sundialslibs[1]),"usr")
srcdir = joinpath(BinDeps.depsdir(sundialslibs[1]),"src",sundialsver)

provides(SimpleBuild,
    (@build_steps begin
        GetSources(sundialslibs[1])
        @build_steps begin
            ChangeDirectory(srcdir)
            `./configure --prefix=$prefix --enable-shared`
            `make install`
        end
    end), sundialslibs)

if enable_sensitivities
@BinDeps.install Dict(:libsundials_cvodes => :libsundials_cvodes,
                  :libsundials_idas => :libsundials_idas,
                  :libsundials_kinsol => :libsundials_kinsol,
                  :libsundials_nvecserial => :libsundials_nvecserial)
else
@BinDeps.install Dict(:libsundials_cvode => :libsundials_cvode,
                  :libsundials_ida => :libsundials_ida,
                  :libsundials_kinsol => :libsundials_kinsol,
                  :libsundials_nvecserial => :libsundials_nvecserial)
end

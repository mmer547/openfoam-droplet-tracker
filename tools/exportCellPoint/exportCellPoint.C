#include "fvCFD.H"
#include "interpolationCellPoint.H"
#include "polyMeshTetDecomposition.H"
#include "timeSelector.H"
#include "volPointInterpolation.H"
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <unistd.h>

using namespace Foam;

static std::ofstream output(const fileName& path)
{
    std::ofstream stream;
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream.open(path.c_str());
    stream << std::setprecision(17);
    return stream;
}

static void writeVector(std::ostream& stream, const vector& value)
{
    stream << value.x() << ' ' << value.y() << ' ' << value.z() << '\n';
}

int main(int argc, char *argv[])
{
    argList::noParallel();
    timeSelector::addOptions(false);
    argList::addOption("samplePoints", "file", "Optional xyz rows for native cellPoint verification");
    #include "setRootCase.H"
    #include "createTime.H"
    const instantList times = timeSelector::select0(runTime, args);
    #include "createMesh.H"
    const fileName dest = runTime.path()/"postProcessing"/"trackerCellPoint";
    const fileName stage = dest + ".tmp-" + Foam::name(::getpid());
    if (exists(dest) || exists(stage))
    {
        FatalErrorInFunction << "Export directory already exists: " << dest
            << " (or " << stage << ')' << exit(FatalError);
    }
    mkDir(stage);
    auto points = output(stage/"points.dat");
    for (const point& p : mesh.points()) writeVector(points, p);
    for (const point& p : mesh.cellCentres()) writeVector(points, p);
    points.close();
    auto volumes = output(stage/"volumes.dat");
    for (const scalar volume : mesh.V()) volumes << volume << '\n';
    volumes.close();
    auto tets = output(stage/"tets.dat");
    forAll(mesh.cells(), celli)
    {
        for (const tetIndices& tet : polyMeshTetDecomposition::cellTetIndices(mesh, celli))
        {
            const triFace ids = tet.faceTriIs(mesh);
            tets << mesh.nPoints()+celli << ' ' << ids[0] << ' ' << ids[1] << ' ' << ids[2] << '\n';
        }
    }
    tets.close();
    mkDir(stage/"patches");
    for (const polyPatch& patch : mesh.boundaryMesh())
    {
        auto faces = output(stage/"patches"/(patch.name() + ".dat"));
        forAll(patch, localFace)
        {
            const label facei = patch.start()+localFace;
            for (const tetIndices& tet : polyMeshTetDecomposition::faceTetIndices
                 (mesh, facei, mesh.faceOwner()[facei]))
            {
                const triFace ids = tet.faceTriIs(mesh);
                faces << ids[0] << ' ' << ids[1] << ' ' << ids[2] << '\n';
            }
        }
        faces.close();
    }
    forAll(times, ti)
    {
        runTime.setTime(times[ti], ti);
        if (mesh.readUpdate() != polyMesh::UNCHANGED)
        {
            FatalErrorInFunction << "Only static meshes are supported" << exit(FatalError);
        }
        const volVectorField U(IOobject("U", runTime.timeName(), mesh, IOobject::MUST_READ,
                                        IOobject::NO_WRITE), mesh);
        const auto pointU = volPointInterpolation::New(mesh).interpolate(U);
        const fileName dir = stage/runTime.timeName();
        mkDir(dir);
        auto velocity = output(dir/"U.dat");
        for (const vector& u : pointU()) writeVector(velocity, u);
        for (const vector& u : U) writeVector(velocity, u);
        velocity.close();
        if (args.found("samplePoints"))
        {
            std::ifstream input(args.get<fileName>("samplePoints").c_str());
            if (!input) throw std::runtime_error("Cannot read samplePoints");
            auto samples = output(dir/"samples.dat");
            interpolationCellPoint<vector> scheme(U);
            point p;
            while (input >> std::ws && !input.eof())
            {
                if (!(input >> p.x() >> p.y() >> p.z()))
                    throw std::runtime_error("Invalid samplePoints row");
                const label celli = mesh.findCell(p);
                if (celli < 0)
                {
                    samples << "-1 -1 -1 -1 0 0 0 0 0 0 0\n";
                    continue;
                }
                const cellPointWeight cpw(mesh, p, celli);
                const triFace& ids = cpw.faceVertices();
                samples << celli << ' ' << ids[0] << ' ' << ids[1] << ' ' << ids[2];
                for (int j=0; j<4; ++j) samples << ' ' << cpw.weights()[j];
                samples << ' ';
                writeVector(samples, scheme.interpolate(cpw));
            }
            samples.close();
        }
    }
    if (std::rename(stage.c_str(), dest.c_str()) != 0)
        throw std::runtime_error("Cannot publish finished export");
    Info << "Wrote " << dest << nl;
    return 0;
}

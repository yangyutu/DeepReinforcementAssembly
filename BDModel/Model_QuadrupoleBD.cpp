/*
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++11']
cfg['linker_args'] = ['-L/opt/OpenBLAS/lib  -llapack -lblas  -pthread -no-pie']
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <memory>
#include <fstream>
#include <string>
#include <cmath>
#include <limits>
#include <sstream>
#include <fcntl.h>
#include <stdint.h>


namespace py = pybind11;
typedef std::vector<double> State;

struct Model_QuadrupoleBD{

    Model_QuadrupoleBD(){}
    Model_QuadrupoleBD(std::string fileTag, int outputFlag, int randomSeed);
    void Initializer(std::string filetag0);
    double callpsi6();
    ~Model_QuadrupoleBD(){ }
    void run(int action);
    void createInitialState();
    void setInitialConfigFile(std::string fileName);
    py::array_t<float> getParticleConfig();
    py::array_t<float> getInfo();
    
    
    

    static const int np = 300;
    static const int np2 = 600;
    const double Os_pressure = 0.02198; 
    int numActions;
    int stateDim;
    /*
     * 0kT   = 0.00000
     * 1kT   = 0.00767
     * 2kT   = 0.01490
     * 2.5kT = 0.01845
     * 3kT   = 0.02198
     * 4kT   = 0.02896
     */
    std::vector<std::vector<int>> nlist;
    double control_dt;
    int R, n_rows, n_cols;
    double dx1, dx2;
    int L;
    int outputTrajFlag;
    int opt, nstep;
    int trajOutputInterval;
    int timeCounter,fileCounter;
    bool trajOutputFlag;
    std::ofstream trajOs, opOs;
    std::string filetag;
    double r[np2], psi6, c6, rg, lambda;
    double Dss;
    std::string initialConfigFile;

    void outputTrajectory(std::ostream& os);
    void outputOrderParameter(std::ostream& os);
    void readxyz(const std::string filename);
    
    void runHelper(int nstep, int opt);
    void buildlist(int);

    int nxyz[np][2];
    double F[np2],D[np2];
    double randisp[np2], dsscalcu[np];
    void forces(int);
    double EMAG(double,double);
    
    void calOp();
    void calDss();
    int ecorrectflag;
    int DG;
    double a, tempr, fcm, kb, rmin,pfpp,kappa,re,rcut,fac1,fac2, dpf;
    double dssmin,dssmax, rgdsmin, delrgdsmin, distmin, deldist;
    int randomSeed;
    double dt;
    
    double reward;
    std::default_random_engine rand_generator;
    std::shared_ptr<std::normal_distribution<double>> rand_normal;

};


Model_QuadrupoleBD::Model_QuadrupoleBD(std::string filetag0, int outputFlag, int randomSeed) {
    this->outputTrajFlag = outputFlag;
    this->randomSeed = randomSeed;
    Initializer(filetag0);
}

void Model_QuadrupoleBD::Initializer(std::string filetag0){
    filetag = filetag0;
    //	we have three low dimensional states psi6, c6, rg
    // nstep 10000 correspond to 1s, every run will run 1s
    nstep = 10000; //steps simulated in 1s
    stateDim = 3; // state is composed of Psi6, C6, rg
    dt = 0.1;
//    dt = 1000.0/nstep; //the length of each step in ms 
    control_dt = 1000; //1000 ms
    nstep = std::round(control_dt/dt);
    numActions = 4; // number of voltages used
    trajOutputInterval = 1; // number of seconds between output
    fileCounter = 0;
    this->rand_generator.seed(this->randomSeed);
    rand_normal = std::make_shared<std::normal_distribution<double>>(0.0, 1.0);
    
    a = 1435.0; // radius of particle
    L = 287.0;
    kb = 1.380658e-23; // Boltzmann constant
    Dss = 0.264; //average dss
    
    for (int i = 0; i < np; i++) {
        nxyz[i][0] = 2 * i;
        nxyz[i][1] = 2 * i + 1;
        nlist.push_back(std::vector<int>());
    }
    
}
void Model_QuadrupoleBD::run(int action) { //run an eqivalent of 1s simulation
    this->opt = action;
    
    if (outputTrajFlag) {
        if (this->timeCounter == 0 || ((this->timeCounter + 1) % trajOutputInterval == 0)) {
            this->outputTrajectory(this->trajOs);
            this->outputOrderParameter(this->opOs);
        }
    }
    // for BD dynamics, every run will simply run 10000 steps, correspond to 1s
    this->runHelper(nstep, opt);
    this->timeCounter++;
}

void Model_QuadrupoleBD::setInitialConfigFile(std::string fileName){
    this->initialConfigFile = fileName;
}

void Model_QuadrupoleBD::createInitialState() {
    std::stringstream FileStr;
    FileStr << this->fileCounter;
//    this->readxyz("./StartMeshgridFolder/startmeshgrid" + FileStr.str() + ".txt");
    this->readxyz(this->initialConfigFile);
    std::stringstream ss;
    std::cout << "model initialize at round " << fileCounter << std::endl;
    ss << this->fileCounter++;
    
    if (outputTrajFlag) {
        if (trajOs.is_open()) trajOs.close();
        if (opOs.is_open()) opOs.close();

        this->trajOs.open(filetag + "xyz_" + ss.str() + ".dat");
        this->opOs.open(filetag + "op" + ss.str() + ".dat");
    }
    this->timeCounter = 0;
    this->calOp();
    this->opt = 0;
    this->lambda = 0;
}

void Model_QuadrupoleBD::outputTrajectory(std::ostream& os) {

    for (int i = 0; i < np; i++) {
        os << i << "\t";
        os << r[2 * i]/a << "\t";
        os << r[2 * i + 1]/a << "\t";
        os << std::endl;
    }
}

void Model_QuadrupoleBD::outputOrderParameter(std::ostream& os) {
    os << this->timeCounter << "\t";
    os << psi6 << "\t";
    os << c6 << "\t";
    os << rg << "\t";
    os << opt << "\t";
    os << lambda << "\t";
    os << std::endl;
}

void Model_QuadrupoleBD::readxyz(const std::string filename) {
    std::ifstream is;
    is.open(filename.c_str());
    std::string line;
    double dum;
    for (int i = 0; i < np; i++) {
        getline(is, line);
        std::stringstream linestream(line);
        linestream >> dum;
        linestream >> r[2 * i];
        linestream >> r[2 * i + 1];
    }
    for (int i = 0; i < np * 2; i++) {
        r[i] *= a;
    }

    is.close();
}


void Model_QuadrupoleBD::runHelper(int nstep, int controlOpt) {

    tempr = 20.0;
    fac1 = 5.9582e7;
    fac2 = 40.5622;
    rcut = 5.0 * a;
    re = 5.0 * a;
    kappa = 1435.0/10;
    pfpp = 2.2975 * a;
    fcm = -0.4667;
    DG = 71.428 * a;
    rmin = 3780;
    rgdsmin = 22250;
    delrgdsmin = -250;
    distmin = 0;
    deldist = 1400;

    dpf = 1;



    if (controlOpt == 0) {
        lambda = 0.219;
    } else if (controlOpt == 1) {
        lambda = 0.8744;
    } else if (controlOpt == 2) {
        lambda = 1.9674;
    } else {
        lambda = 19.73;
    }


    fac1 = fac1 / a;
    fac2 = fac2 * sqrt((273 + tempr) / a);
    fac2 = fac2 / sqrt(dt);

    int step = 0;
    while (step < nstep) {
        for (int j = 0; j < np; j++) {
            for (int k = 0; k < 2; k++) {
                
                double randTemp = (*rand_normal)(rand_generator);
                randisp[nxyz[j][k]] =  randTemp * sqrt(1.0 / this->Dss);
            }
        }

        forces(step);
        double u;
        for (int j = 0; j < np2; j++) {
            u = this->Dss * (F[j] * fac1+randisp[j]* fac2);
            r[j] += u * dt;
        }
        step++;
    }
    this->calOp();
}

void Model_QuadrupoleBD::forces(int sstep) {
    double RX, RY, EMAGI, EMAGJ, dE2x, dE2y, Fdepx, Fdepy;
    double STEP = 1e-3;
    double rij[2];
    double Fpp, Fhw, Exi, Exj, Eyi,Eyj,Ezi,Ezj, FOS;
    Fhw = 0.417;
    double Fo = 1e18*0.75*lambda*kb*(273+tempr)/a;
    for (int i = 0 ; i < np2; i++) {F[i] = 0.0;} 
    for (int i = 0; i < np-1; i++){
        Exi = -4.0*r[nxyz[i][0]]/DG;
        Eyi = 4.0*r[nxyz[i][1]]/DG;

	if ((sstep+1)%100 == 0 || sstep ==0 ){
	    buildlist(i); //
	} 
	int nlistsize = nlist[i].size();
        for (int jj = 0; jj < nlistsize; jj++){
	    int j = nlist[i][jj];
            rij[0] = r[nxyz[j][0]] - r[nxyz[i][0]];
            rij[1] = r[nxyz[j][1]] - r[nxyz[i][1]];
	    double rijsep = sqrt(rij[0]*rij[0] + rij[1]*rij[1]);
            if (rijsep < 2*a){ //overlap
                Fpp = Fhw;

            } else if( rijsep < rcut) {
                Fpp = 1e18*kb*(tempr+273)*kappa*pfpp*exp(-kappa*(rijsep - 2.0*a)/a)/a;
            } else {
                Fpp = 0;
            }
            
	    if (rijsep > 2*a && (rijsep < (2*a + 2*L))){
		FOS = (4.0/3.0)* Os_pressure*M_PI*(-0.75*(a+L)*(a+L)*1e-18+0.1875*rijsep*rijsep*1e-18)*1e9;
//		FOS = 0;
	    } else {
		FOS = 0;
	    }
// particle-particle interaction          
            
            
            F[nxyz[i][0]] = F[nxyz[i][0]]  - (Fpp+FOS)*rij[0]/rijsep;
            F[nxyz[i][1]] = F[nxyz[i][1]]  - (Fpp+FOS)*rij[1]/rijsep;
            
            F[nxyz[j][0]] = F[nxyz[j][0]]  + (Fpp+FOS)*rij[0]/rijsep;
            F[nxyz[j][1]] = F[nxyz[j][1]]  + (Fpp+FOS)*rij[1]/rijsep;
        }
    }
//  field interaction
    for (int i = 0; i < np; i++) {
        RX = r[nxyz[i][0]];
        RY = r[nxyz[i][1]];
        EMAGI = EMAG(RX, RY);
        
        RX = r[nxyz[i][0]] + STEP;
        RY = r[nxyz[i][1]];
        EMAGJ = EMAG(RX, RY);
        
        dE2x = (EMAGJ * EMAGJ - EMAGI * EMAGI) / STEP;
        RX = r[nxyz[i][0]];
        RY = r[nxyz[i][1]] + STEP;

        EMAGJ = EMAG(RX, RY);
        dE2y = (EMAGJ * EMAGJ - EMAGI * EMAGI) / STEP;
        Fdepx = (2 * 1e18 * kb * (tempr + 273) * lambda / fcm) * dE2x;
        Fdepy = (2 * 1e18 * kb * (tempr + 273) * lambda / fcm) * dE2y;
        F[nxyz[i][0]] += Fdepx;
        F[nxyz[i][1]] += Fdepy;
    }
}

double Model_QuadrupoleBD::EMAG(double RX, double RY) {
    double RT = sqrt(RX * RX + RY * RY);
    double result = 4 * RT / DG;
    return result;
}

void Model_QuadrupoleBD::calOp() {

    int nb[np], con[np];
    double rx[np], ry[np];
    double rxij, ryij, theta, psir[np], psii[np], numer, denom, testv, ctestv;
    double rgmean, xmean, ymean, accumpsi6r, accumpsi6i;
    ctestv = 0.32;
    for (int i = 0; i < np; i++) {
        rx[i] = r[nxyz[i][0]];
        ry[i] = r[nxyz[i][1]];
    }

    for (int i = 0; i < np; i++) {
        nb[i] = 0;
        psir[i] = 0.0;
        psii[i] = 0.0;
        for (int j = 0; j < np; j++) {
            if (i != j) {
                rxij = rx[j] - rx[i];
                ryij = ry[j] - ry[i];
                double RP = sqrt(rxij * rxij + ryij * ryij);
                if (RP < rmin) {
                    nb[i] += 1;
                    theta = std::atan2(ryij, rxij);
                    psir[i] += cos(6 * theta);
                    psii[i] += sin(6 * theta);
                }
            }
                      
        } 
            if (nb[i] > 0) {
                psir[i] /=  nb[i];
                psii[i] /=  nb[i];
            }
    }
    this-> psi6 = 0;
    accumpsi6r = 0;
    accumpsi6i = 0;
    for (int i = 0; i < np; i++) {

        accumpsi6r = accumpsi6r + psir[i];
        accumpsi6i = accumpsi6i + psii[i];
    }
    accumpsi6r = accumpsi6r / np;
    accumpsi6i = accumpsi6i / np;
    this-> psi6 = sqrt(accumpsi6r * accumpsi6r + accumpsi6i * accumpsi6i);
    c6 = 0.0;

    for (int i = 0; i < np; i++) {
        con[i] = 0;
        for (int j = 0; j < np; j++) {
            rxij = rx[j] - rx[i];
            ryij = ry[j] - ry[i];
            double rp = sqrt(rxij * rxij + ryij * ryij);
            if ((i != j)&&(rp <= rmin)) {

                numer = psir[i] * psir[j] + psii[i] * psii[j];
                double temp = psii[i] * psir[j] - psii[j] * psir[i];
                denom = sqrt(numer * numer + temp*temp);
                testv = numer / denom;
                if (testv >= ctestv) {
                    con[i] += 1;
                }
            }
        }
        c6 = c6 + con[i];
    }

    c6 /= np;
    //      calculate Rg
    xmean = 0;
    ymean = 0;

    for (int i = 0; i < np; i++) {
        xmean = xmean + rx[i];
        ymean = ymean + ry[i];
    }
    xmean /= np;
    ymean /= np;


    rgmean = 0;
    for (int i = 0; i < np; i++) {
        rgmean = rgmean + (rx[i] - xmean)*(rx[i] - xmean);
        rgmean = rgmean + (ry[i] - ymean)*(ry[i] - ymean);
    }
    rgmean /= np;

    rgmean = sqrt(rgmean);
    rg = rgmean;

}

void Model_QuadrupoleBD::buildlist(int i){
    nlist[i].clear();
    for (int kk = i+1; kk < np; kk++){
	double rijx = r[nxyz[kk][0]] - r[nxyz[i][0]];
	double rijy = r[nxyz[kk][1]] - r[nxyz[i][1]];
	double rijsep = sqrt(rijx*rijx+rijy*rijy);
	if (rijsep < rcut){
	    nlist[i].push_back(kk);
	}
    }
}


py::array_t<float> Model_QuadrupoleBD::getParticleConfig(){
    
    float out[np2];
    for(int i = 0; i < np2; i++)
    {
        //out[i] = this->r[i];
        out[i] = this->r[i] / a;
    }    
    py::array_t<float> output(np2, out);    
    return output;
}

py::array_t<float> Model_QuadrupoleBD::getInfo(){
    float out[5];
    out[0]= psi6;
    out[1]= c6;
    out[2]= rg;
    out[3]= opt;
    out[4]= lambda;
    py::array_t<float> output(5, out);
    
    return output;
}


PYBIND11_MODULE(Model_QuadrupoleBD, m) {    
    py::class_<Model_QuadrupoleBD>(m, "Model_QuadrupoleBD")
        .def(py::init<std::string, int, int>())
        .def("setInitialConfigFile", &Model_QuadrupoleBD::setInitialConfigFile)
        .def("createInitialState", &Model_QuadrupoleBD::createInitialState)
    	.def("run", &Model_QuadrupoleBD::run)
        .def("getParticleConfig", &Model_QuadrupoleBD::getParticleConfig)
        .def("getInfo", &Model_QuadrupoleBD::getInfo);
}
//
// Created by lab on 2022/11/10.
//
#include "ICP.h"
namespace ICP {
/// Weight functions
/// @param Residuals
/// @param Parameter
void uniform_weight(Eigen::VectorXd& r) {
    r = Eigen::VectorXd::Ones(r.rows());
}
/// @param Residuals
/// @param Parameter
void pnorm_weight(Eigen::VectorXd& r, double p, double reg = 1e-8) {
    for (int i = 0; i<r.rows(); ++i) {
        r(i) = p / (std::pow(r(i), 2 - p) + reg);
    }
}
/// @param Residuals
/// @param Parameter
void tukey_weight(Eigen::VectorXd& r, double p) {
    for (int i = 0; i<r.rows(); ++i) {
        if (r(i) > p) r(i) = 0.0;
        else r(i) = std::pow((1.0 - std::pow(r(i) / p, 2.0)), 2.0);
    }
}
/// @param Residuals
/// @param Parameter
void fair_weight(Eigen::VectorXd& r, double p) {
    for (int i = 0; i<r.rows(); ++i) {
        r(i) = 1.0 / (1.0 + r(i) / p);
    }
}
/// @param Residuals
/// @param Parameter
void logistic_weight(Eigen::VectorXd& r, double p) {
    for (int i = 0; i<r.rows(); ++i) {
        r(i) = (p / r(i))*std::tanh(r(i) / p);
    }
}

/// @param Residuals
/// @param Parameter
void trimmed_weight(Eigen::VectorXd& r, double p) {
    std::vector<std::pair<int, double> > sortedDist(r.rows());
    for (int i = 0; i<r.rows(); ++i) {
        sortedDist[i] = std::pair<int, double>(i, r(i));
    }
    std::sort(sortedDist.begin(), sortedDist.end(), sort_pred());
    r.setZero();
    int nbV = r.rows()*p;
    for (int i = 0; i<nbV; ++i) {
        r(sortedDist[i].first) = 1.0;
    }
}
/// @param Residuals
/// @param Parameter
void welsch_weight(Eigen::VectorXd& r, double p) {
    for (int i = 0; i<r.rows(); ++i) {
        r(i) = std::exp(-r(i)*r(i)/(2*p*p));
//            energy += 1.0 - std::exp(-r(i)*r(i)/(2*p*p));
    }
}

/// @param Residuals
/// @param Parameter
void autowelsch_weight(Eigen::VectorXd& r, double p) {
    double median;
    igl::median(r, median);
    welsch_weight(r, p*median/(std::sqrt(2)*2.3));
    //welsch_weight(r,p);
}

/// Energy functions
/// @param Residuals
/// @param Parameter
double uniform_energy(Eigen::VectorXd& r) {
    double energy = 0;
    for (int i = 0; i<r.rows(); ++i) {
        energy += r(i)*r(i);
    }
    return energy;
}
/// @param Residuals
/// @param Parameter
double pnorm_energy(Eigen::VectorXd& r, double p, double reg = 1e-8) {
    double energy = 0;
    for (int i = 0; i<r.rows(); ++i) {
        energy += (r(i)*r(i))*p / (std::pow(r(i), 2 - p) + reg);
    }
    return energy;
}
/// @param Residuals
/// @param Parameter
double tukey_energy(Eigen::VectorXd& r, double p) {
    double energy = 0;
    double w;
    for (int i = 0; i<r.rows(); ++i) {
        if (r(i) > p) w = 0.0;
        else w = std::pow((1.0 - std::pow(r(i) / p, 2.0)), 2.0);

        energy += (r(i)*r(i))*w;
    }
    return energy;
}
/// @param Residuals
/// @param Parameter
double fair_energy(Eigen::VectorXd& r, double p) {
    double energy = 0;
    for (int i = 0; i<r.rows(); ++i) {
        energy += (r(i)*r(i))*1.0 / (1.0 + r(i) / p);
    }
    return energy;
}
/// @param Residuals
/// @param Parameter
double logistic_energy(Eigen::VectorXd& r, double p) {
    double energy = 0;
    for (int i = 0; i<r.rows(); ++i) {
        energy += (r(i)*r(i))*(p / r(i))*std::tanh(r(i) / p);
    }
    return energy;
}
/// @param Residuals
/// @param Parameter
double trimmed_energy(Eigen::VectorXd& r, double p) {
    std::vector<std::pair<int, double> > sortedDist(r.rows());
    for (int i = 0; i<r.rows(); ++i) {
        sortedDist[i] = std::pair<int, double>(i, r(i));
    }
    std::sort(sortedDist.begin(), sortedDist.end(), sort_pred());
    Eigen::VectorXd t = r;
    t.setZero();
    double energy = 0;
    int nbV = r.rows()*p;
    for (int i = 0; i<nbV; ++i) {
        energy += r(i)*r(i);
    }
    return energy;
}

/// @param Residuals
/// @param Parameter
double welsch_energy(Eigen::VectorXd& r, double p) {
    double energy = 0;
    for (int i = 0; i<r.rows(); ++i) {
        energy += 1.0 - std::exp(-r(i)*r(i)/(2*p*p));   //论文中的公式(14)
    }
    return energy;
}
/// @param Residuals
/// @param Parameter
double autowelsch_energy(Eigen::VectorXd& r, double p) {
    double energy = 0;
    energy = welsch_energy(r, 0.5);
    return energy;
}
/// @param Function type
/// @param Residuals
/// @param Parameter
void robust_weight(Function f, Eigen::VectorXd& r, double p) {
    switch (f) {
        case PNORM: pnorm_weight(r, p); break;
        case TUKEY: tukey_weight(r, p); break;
        case FAIR: fair_weight(r, p); break;
        case LOGISTIC: logistic_weight(r, p); break;
        case TRIMMED: trimmed_weight(r, p); break;
        case WELSCH: welsch_weight(r, p); break;
        case AUTOWELSCH: autowelsch_weight(r,p); break;
        case NONE: uniform_weight(r); break;
        default: uniform_weight(r); break;
    }
}

//Cacl energy
double get_energy(Function f, Eigen::VectorXd& r, double p) {
    double energy = 0;
    switch (f) {
        //case PNORM: pnorm_weight(r,p); break;
        case TUKEY: energy = tukey_energy(r, p); break;
        case FAIR: energy = fair_energy(r, p); break;
        case LOGISTIC: energy = logistic_energy(r, p); break;
        case TRIMMED: energy = trimmed_energy(r, p); break;
        case WELSCH: energy = welsch_energy(r, p); break;
        case AUTOWELSCH: energy = autowelsch_energy(r, p); break;
        case NONE: energy = uniform_energy(r); break;
        default: energy = uniform_energy(r); break;
    }
    return energy;
}
}

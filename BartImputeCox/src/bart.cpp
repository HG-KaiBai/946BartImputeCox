/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017 Robert McCulloch and Rodney Sparapani
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */
#include "bart.h"

// Modify the Setdata and the overloaded operator= functions to copy bart between objects.

//--------------------------------------------------
//constructor
bart::bart():m(200),t(m),pi(),p(0),n(0),x(0),y(0),xi(),allfit(0),r(0),ftemp(0),di(),dartOn(false) {}
bart::bart(size_t im):m(im),t(m),pi(),p(0),n(0),x(0),y(0),xi(),allfit(0),r(0),ftemp(0),di(),dartOn(false) {}
bart::bart(const bart& ib):m(ib.m),t(m),pi(ib.pi),p(ib.p),n(ib.n),x(ib.x),y(ib.y),xi(ib.xi),allfit(0),r(0),ftemp(0),di(ib.di),dartOn(ib.dartOn) {
    if (ib.allfit) {
        allfit = new double[n];
        std::copy(ib.allfit, ib.allfit + n, allfit);
    }
    if (ib.r) {
        r = new double[n];
        std::copy(ib.r, ib.r + n, r);
    }
    if (ib.ftemp) {
        ftemp = new double[n];
        std::copy(ib.ftemp, ib.ftemp + n, ftemp);
    }
}
bart::~bart()
{
   if(allfit) delete[] allfit;
   if(r) delete[] r;
   if(ftemp) delete[] ftemp;
}

//--------------------------------------------------
//operators
bart& bart::operator=(const bart& rhs) {    // Copy all data members from rhs to this and free the memory of the old data members.
    if (&rhs != this) {
        m = rhs.m;
        t = rhs.t;
        pi = rhs.pi;
        p = rhs.p;
        n = rhs.n;
        x = rhs.x;
        y = rhs.y;
        xi = rhs.xi;
        di = rhs.di;
        dartOn = rhs.dartOn;

        if (allfit) delete[] allfit;
        if (rhs.allfit) {
            allfit = new double[n];
            std::copy(rhs.allfit, rhs.allfit + n, allfit);
        } else {
            allfit = nullptr;
        }

        if (r) delete[] r;
        if (rhs.r) {
            r = new double[n];
            std::copy(rhs.r, rhs.r + n, r);
        } else {
            r = nullptr;
        }

        if (ftemp) delete[] ftemp;
        if (rhs.ftemp) {
            ftemp = new double[n];
            std::copy(rhs.ftemp, rhs.ftemp + n, ftemp);
        } else {
            ftemp = nullptr;
        }
    }
    return *this;
}
//--------------------------------------------------
//get,set
//set the number of trees
void bart::setm(size_t m)
{
   t.resize(m);
   this->m = t.size();

   if(allfit && (xi.size()==p)) predict(p,n,x,allfit);
}

//--------------------------------------------------
// Set the training data X to choose the splitting value.
void bart::setxinfo(xinfo& _xi)
{
   size_t p=_xi.size();
   xi.resize(p);
   for(size_t i=0;i<p;i++) {
     size_t nc=_xi[i].size();
      xi[i].resize(nc);
      for(size_t j=0;j<nc;j++) xi[i][j] = _xi[i][j];
   }
}
//--------------------------------------------------
// Set the training data X and Y, and the correspoing dimensions with fitted value
void bart::setdata(size_t p, size_t n, double *x, double *y, size_t numcut)
{
  int* nc = new int[p];
  for(size_t i=0; i<p; ++i) nc[i]=numcut;
  this->setdata(p, n, x, y, nc);
  delete [] nc;
}

void bart::setdata(size_t p, size_t n, double *x, double *y, int *nc)
{
   this->p=p; this->n=n; this->x=x; this->y=y;
   if(xi.size()==0) makexinfo(p,n,&x[0],xi,nc);
   // Clear and reload everything related to the past training data
   if(allfit) delete[] allfit;
   allfit = new double[n];
   predict(p,n,x,allfit);

   if(r) delete[] r;
   r = new double[n];

   if(ftemp) delete[] ftemp;
   ftemp = new double[n];

   di.n=n; di.p=p; di.x = &x[0]; di.y=r;

   nv.clear();
   pv.clear();
   for(size_t j=0;j<p;j++){
     nv.push_back(0);
     pv.push_back(1/(double)p);
   }
}
//--------------------------------------------------
// Predict the results for the test data X and store the results in the vector fp
void bart::predict(size_t p, size_t n, double *x, double *fp)
//uses: m,t,xi
{
   double *fptemp = new double[n];

   for(size_t j=0;j<n;j++) fp[j]=0.0;
   for(size_t j=0;j<m;j++) {
      fit(t[j],xi,p,n,x,fptemp);
      for(size_t k=0;k<n;k++) fp[k] += fptemp[k];
   }
   delete[] fptemp;

}
//--------------------------------------------------
// For all tree, draw the new structure and parameters
void bart::draw(double sigma, rn& gen)
{
   for(size_t j=0;j<m;j++) {
      fit(t[j],xi,p,n,x,ftemp);
      for(size_t k=0;k<n;k++) {
         allfit[k] = allfit[k]-ftemp[k];
         r[k] = y[k]-allfit[k];
      }
      bd(t[j],xi,di,pi,sigma,nv,pv,aug,gen);
      drmu(t[j],xi,di,pi,sigma,gen);
      fit(t[j],xi,p,n,x,ftemp);
      for(size_t k=0;k<n;k++) allfit[k] += ftemp[k];
   }
   if(dartOn) {
     draw_s(nv,lpv,theta,gen);
     draw_theta0(const_theta,theta,lpv,a,b,rho,gen);
     for(size_t j=0;j<p;j++) pv[j]=::exp(lpv[j]);
   }
}
//--------------------------------------------------
//public functions
void bart::pr() //print to screen
{
   cout << "*****bart object:\n";
   cout << "m: " << m << std::endl;
   cout << "t[0]:\n " << t[0] << std::endl;
   cout << "t[m-1]:\n " << t[m-1] << std::endl;
   cout << "prior and mcmc info:\n";
   pi.pr();
   if(dart){
     cout << "*****dart prior (On):\n";
     cout << "a: " << a << std::endl;
     cout << "b: " << b << std::endl;
     cout << "rho: " << rho << std::endl;
     cout << "augmentation: " << aug << std::endl;
   }
   else cout << "*****dart prior (Off):\n";
   if(p) cout << "data set: n,p: " << n << ", " << p << std::endl;
   else cout << "data not set\n";
}

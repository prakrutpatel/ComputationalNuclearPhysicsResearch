
# Derivation of an optical potential for statically deformed nuclei from a global spherical potential

This project is a result of my summer research done at Eckerd College in collaboration with [Prof. Stephen Weppner](https://scholar.google.com/citations?user=UKnDgu4AAAAJ&hl=en&oi=ao). This project aims to describe the recovery of a nuclei after being statically deformed, i.e. after a collision. The aim of this project was to match our results with real life data gathered from Brookhaven National Lab. Currently there exist multiple theories that decribe this so called recovery process relatively well but their data is contrained to a certain range of nuclei, our research was meant to take out this contrain so that all nuclei behavior can be explained using 1 theory.


Although this project is meant for use by me and Prof. Weppner, it could be used by anyone to simulate the results.


## Research

The coupled-channel theory is a way of treating nonelastic channels, in particular those arising from collective excitations characterized by nuclear deformations. Proper treatment of such excitations is required to provide an accurate description of experimental nuclear-reaction data and predict a variety of scattering observables. In this research, we explore the possibility of generalizing an optical model potential that can be used in coupled-channel calculation for a wide range of nuclei targets. To achieve this, we will start with the Koning-Delaroche global spherical potential for nucleon projectiles (protons and neutrons) and modify it systematically based on the nuclei’s physical properties. In order to simulate scattering, we use ECIS-12, a Fortran program written in the ’90s that implements coupled-channel theory in a Schrodinger-Dirac context. We get scattering observables such as cross-sections and spin polarization as output from the program. Predictions from our model for these observables are in reasonable agreement with the experimental data. These results suggest that our deformed Koning-Delaroche potential provides a useful optical model potential for the statically deformed nuclei.
## Work

This repo contains all codes and files used during my research with Prof. Stephen Weppner at Eckerd College. I wrote all the codes present in this project which are used to create optical potential files, perform data extraction and data manipulation, data analysis, case-by-case parameter switching and data plot creation for scientific journals. All of these files were created using python and at the backbone of our research was ECIS-12 program written in Fortran by - Service de Physique Theorique, Laboratoire de la Direction des Sciences de la Matiere du Commissariat a l'Energie Atomique, CE-Saclay, F-91190 Gif-sur-Yvette CEDEX, France through the Nuclear Energy Agency Data Bank, Issy-les-Moulineaux, France. These programs collectively were used in a pipeline process that was controlled by a bash script which would control the flow of our programs and perform checks on the data during the process. Multithreading was implemented to increase the speed of our calculations. Neutron and proton based codes were merged to increase effeciency.

Our goal was to come up with a equation that could explain the recovery of an atom after deformation, this equation should match the experimental data across a huge range of elements with varying energy levels, while perform more than adequate recovery of atoms. In the end our result correlated extremely well with the experimental data and Prof. Weppner is in the process of writting up a paper.


## Presentation

My Undergrad Symposium Presentation on this project can be found at - [Symposium Presentation](https://physics.prakrut.dev/)
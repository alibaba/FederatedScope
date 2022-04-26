#!/bin/bash

jupyter nbconvert --to notebook --execute --inplace notebooks/01_quick_start.ipynb
return=$?
if [ $return = 1 ];
then
    exit 1
fi


jupyter nbconvert --to notebook --execute --inplace notebooks/02_start_your_own_case.ipynb
return=$?
if [ $return = 1 ];
then
    exit 1
fi

jupyter nbconvert --to notebook --execute --inplace notebooks/03_personalized_FL.ipynb
return=$?
if [ $return = 1 ];
then
    exit 1
else
    exit 0
fi


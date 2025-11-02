#pragma once

#include <math.h>
#include <random>

#include <iostream>
#include <stdio.h>
#include <string>
#include "time.h"
#include "Windows.h"

using namespace std;

namespace Generator {
	int CountGPUs();
	string* GetGPUNames();
	uint32_t* Generate(uint32_t* startingPoint, int totalPoints);
}
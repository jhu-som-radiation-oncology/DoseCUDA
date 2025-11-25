#include <algorithm>

//#define PY_SSIZE_T_CLEAN
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <object.h>
#include <numpy/arrayobject.h>

#include "IMRTClasses.cuh"
#include "IMPTClasses.cuh"
#include "MemoryClasses.h"


/** @brief Check a NumPy array for dimensionality and contained element type
 * 	@param arr
 * 		NumPy array pointer
 * 	@param dim
 * 		Required dimensionality
 * 	@param type
 * 		Required type constant
 * 	@returns true if the constraint is satisfied, false if not
 */
static bool pyarray_typecheck(const PyArrayObject *arr, int dim, int type) {

	return PyArray_NDIM(arr) == dim && PyArray_TYPE(arr) == type;
}


/** @brief Fetch the array's data pointer as a pointer to T */
template <class T>
static T *pyarray_as(PyArrayObject *arr) {

	return reinterpret_cast<T *>(PyArray_DATA(arr));
}


/** @brief Borrow a reference to a NumPy array object contained by a class
 * 	@param self
 * 		Class containing the array
 * 	@param attr
 * 		Member/field name of the array
 * 	@param dim
 * 		Dimensionality/rank of the array
 * 	@param[out] arr
 * 		The `PyArrayObject *` will be written here on success. You do not need
 * 		to `Py_DECREF` this object
 * 	@returns true on success, false on error. If an error occurs, a Python
 * 		exception will already have been raised
 */
static bool pyobject_getarray(PyObject *self, const char *attr, int dim, PyArrayObject **arr) {

	PyObject *ptr = PyObject_GetAttrString(self, attr);
	if (!ptr) {
		return false;
	}

	bool result = false;
	*arr = reinterpret_cast<PyArrayObject *>(ptr);
	if (PyArray_Check(ptr) && pyarray_typecheck(*arr, dim, NPY_FLOAT)) {
		result = true;
	} else {
		PyErr_Format(PyExc_ValueError, "'%s' must be %d-dimensional and of type float.", attr, dim);
	}
	Py_DECREF(ptr);
	return result;
}


/** @brief Get a `double` from a Python class
 * 	@param self
 * 		Class
 * 	@param attr
 * 		Member/field name of the float
 * 	@param[out] value
 * 		The result will be written here on success
 * 	@returns true on success, false on error. If an error occurs, a Python
 * 		exception will have been raised
 */
static bool pyobject_getfloat(PyObject *self, const char *attr, double *value) {

	PyObject *ptr = PyObject_GetAttrString(self, attr);
	if (!ptr) {
		return false;
	}
	*value = PyFloat_AsDouble(ptr);
	Py_DECREF(ptr);
	return *value == -1.0 ? !PyErr_Occurred() : true;
}


static bool pyobject_getbool(PyObject *self, const char *attr, bool *value) {

	PyObject *ptr = PyObject_GetAttrString(self, attr);
	if (!ptr) {
		return false;
	}
	int result = PyObject_IsTrue(ptr);
	Py_DECREF(ptr);
	*value = result > 0;
	return result >= 0;
}


static PyObject* proton_raytrace(PyObject *self, PyObject *args) {

	PyObject *model_instance, *volume_instance, *beam_instance;
	int gpu_id;	

	// parse arguments
    if (!PyArg_ParseTuple(args, "OOOi", &model_instance, &volume_instance, &beam_instance, &gpu_id))
        return NULL;

	// check beam model properties 
	double vsadx, vsady;

	if (!pyobject_getfloat(model_instance, "VSADX", &vsadx)
	 || !pyobject_getfloat(model_instance, "VSADY", &vsady)) {
		return NULL;
	}

	// check volume data properties
	PyArrayObject *density_array, 
		*spacing_array, 
		*origin_array;
	if (!pyobject_getarray(volume_instance, "voxel_data", 3, &density_array)
	 || !pyobject_getarray(volume_instance, "spacing", 1, &spacing_array)
	 || !pyobject_getarray(volume_instance, "origin", 1, &origin_array)) {
		return NULL;
	}

	// check beam properties
	PyArrayObject *iso_array;
	if (!pyobject_getarray(beam_instance, "iso", 1, &iso_array)) {
		return NULL;
	}

	double ga, ta;
	if (!pyobject_getfloat(beam_instance, "gantry_angle", &ga)
	 || !pyobject_getfloat(beam_instance, "couch_angle", &ta)) {
		return NULL;
	}

	float * spacing = pyarray_as<float>(spacing_array);
	float * origin = pyarray_as<float>(origin_array);
	float * iso = pyarray_as<float>(iso_array);
	double voxel_sp = (double)spacing[0];

	float adjusted_gantry_angle = fmodf(ga + 180.0f, 360.0f);

	try {

		float adjusted_ga = fmodf(ga + 180.0f, 360.0f);

		size_t dims[3] = {
			(size_t)PyArray_DIMS(density_array)[0],
			(size_t)PyArray_DIMS(density_array)[1],
			(size_t)PyArray_DIMS(density_array)[2],
		};

		float adjusted_isocenter[3] = {
			iso[0] - origin[0],
			iso[1] - origin[1],
			iso[2] - origin[2]
		};

		// beam model object
		auto model = IMPTBeam::Model();
		model.vsadx = vsadx;
		model.vsady = vsady;

		// beam object
		IMPTBeam beam_obj = IMPTBeam(adjusted_isocenter, adjusted_ga, ta, &model);

		// dose object
		IMPTDose dose_obj = IMPTDose(dims, voxel_sp);
		HostPointer<float> WETArray(dose_obj.num_voxels);
		dose_obj.DensityArray = pyarray_as<float>(density_array);
		dose_obj.WETArray = WETArray.get();

		//perform raytrace
		proton_raytrace_cuda(gpu_id, &dose_obj, &beam_obj);

		PyObject *return_wet = PyArray_SimpleNewFromData(3, PyArray_DIMS(density_array), PyArray_TYPE(density_array), WETArray.release());

		PyArray_ENABLEFLAGS((PyArrayObject*) return_wet, NPY_ARRAY_OWNDATA);

		return return_wet;

	} catch (std::bad_alloc &) {

		PyErr_SetString(PyExc_MemoryError, "Not enough host memory");

	} catch (std::runtime_error &e) {

		PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", e.what());

	}

	return NULL;

}


static bool spot_compare(const Spot &a, const Spot &b)
{
	return a.energy_id < b.energy_id;
}


/** Create the authoritative spot array from spot data in any order */
static void make_spot_array(PyArrayObject *spots, HostPointer<Spot> &res)
{
	const size_t count = PyArray_DIM(spots, 0);
	const float *src = pyarray_as<float>(spots);

	for (size_t i = 0; i < count; i++) {
		res[i].x = src[4 * i];
		res[i].y = src[4 * i + 1];
		res[i].mu = src[4 * i + 2];
		res[i].energy_id = static_cast<int>(src[4 * i + 3]);
		// printf("Spot %d: x: %f, y: %f, mu: %f, energy_id: %d\n", i, res[i].x, res[i].y, res[i].mu, res[i].energy_id);
	}

	std::sort(&res[0], &res[count], spot_compare);
}


static void make_mlc_array(PyArrayObject *mlc, HostPointer<MLCPair> &res)
{
	const size_t count = PyArray_DIM(mlc, 0);
	const float *src = pyarray_as<float>(mlc);

	for (size_t i = 0; i < count; i++) {
		res[i].x1 = src[i];
		res[i].x2 = src[i + count];
		res[i].y_offset = src[i + 2 * count];
		res[i].y_width = src[i + 3 * count];
		// printf("MLC Pair %d: x1: %f, x2: %f, y_offset: %f, y_width: %f\n", i, res[i].x1, res[i].x2, res[i].y_offset, res[i].y_width);
	}
}


static PyObject* proton_spot(PyObject *self, PyObject *args) {

	PyObject *model_instance, *volume_instance, *wet_instance, *beam_instance;
	int gpu_id;	

	// parse arguments
    if (!PyArg_ParseTuple(args, "OOOOi", &model_instance, &volume_instance, &wet_instance, &beam_instance, &gpu_id))
        return NULL;

	// check beam model properties 
	double vsadx, vsady;

	if (!pyobject_getfloat(model_instance, "VSADX", &vsadx)
	 || !pyobject_getfloat(model_instance, "VSADY", &vsady)) {
		return NULL;
	}

	PyArrayObject *lut_depths_array, *lut_sigmas_array, *lut_idds_array, *lut_divergence_params_array;
	if (!pyobject_getarray(model_instance, "divergence_params", 2, &lut_divergence_params_array)
	|| !pyobject_getarray(model_instance, "lut_depths", 2, &lut_depths_array)
	|| !pyobject_getarray(model_instance, "lut_sigmas", 2, &lut_sigmas_array)
	|| !pyobject_getarray(model_instance, "lut_idds", 2, &lut_idds_array)) {
		return NULL;
	}

	// check volume data properties
	PyArrayObject *density_array, 
		*spacing_array, 
		*origin_array;
	if (!pyobject_getarray(volume_instance, "voxel_data", 3, &density_array)
	 || !pyobject_getarray(volume_instance, "spacing", 1, &spacing_array)
	 || !pyobject_getarray(volume_instance, "origin", 1, &origin_array)) {
		return NULL;
	}

	// check WET data properties
	PyArrayObject *wet_array;
	if (!pyobject_getarray(wet_instance, "voxel_data", 3, &wet_array)) {
		return NULL;
	}

	// check beam properties
	PyArrayObject *iso_array, *spots_array;
	if (!pyobject_getarray(beam_instance, "iso", 1, &iso_array)
	 || !pyobject_getarray(beam_instance, "spot_list", 2, &spots_array)) {
		return NULL;
	}

	double ga, ta;
	if (!pyobject_getfloat(beam_instance, "gantry_angle", &ga)
	 || !pyobject_getfloat(beam_instance, "couch_angle", &ta)) {
		return NULL;
	}

	float * spacing = pyarray_as<float>(spacing_array);
	float * origin = pyarray_as<float>(origin_array);
	float * iso = pyarray_as<float>(iso_array);
	double voxel_sp = (double)spacing[0];

	try {

		float adjusted_ga = fmodf(ga + 180.0f, 360.0f);
		size_t n_energies = PyArray_DIM(lut_depths_array, 0);
		size_t n_spots = PyArray_DIM(spots_array, 0);

		size_t dims[3] = {
			(size_t)PyArray_DIMS(wet_array)[0],
			(size_t)PyArray_DIMS(wet_array)[1],
			(size_t)PyArray_DIMS(wet_array)[2],
		};

		float adjusted_isocenter[3] = {
			iso[0] - origin[0],
			iso[1] - origin[1],
			iso[2] - origin[2]
		};

		IMPTDose dose_obj = IMPTDose(dims, voxel_sp);
		HostPointer<float> DoseArray(dose_obj.num_voxels);

		dose_obj.DoseArray = DoseArray.get();
		dose_obj.DensityArray = pyarray_as<float>(density_array);
		dose_obj.WETArray = pyarray_as<float>(wet_array);

		// beam model object
		auto model = IMPTBeam::Model();
		model.vsadx = vsadx;
		model.vsady = vsady;

		// beam object
		IMPTBeam beam_obj = IMPTBeam(adjusted_isocenter, adjusted_ga, ta, &model);

		HostPointer<Layer> LayerArray(n_energies);
		HostPointer<Spot> SpotArray(n_spots);

		make_spot_array(spots_array, SpotArray);

		beam_obj.n_energies = n_energies;
		beam_obj.layers = LayerArray.get();
		beam_obj.spots = SpotArray.get();
		beam_obj.n_spots = n_spots;
		beam_obj.divergence_params = pyarray_as<float>(lut_divergence_params_array);
		beam_obj.dvp_len = 5;	// R80, energy, quadratic coefficients
		beam_obj.lut_depths = pyarray_as<float>(lut_depths_array);
		beam_obj.lut_sigmas = pyarray_as<float>(lut_sigmas_array);
		beam_obj.lut_idds = pyarray_as<float>(lut_idds_array);
		beam_obj.lut_len = LUT_LENGTH;	// This can now be changed at runtime

		beam_obj.importLayers();

		//compute dose
		proton_spot_cuda(gpu_id, &dose_obj, &beam_obj);

		PyObject *return_dose = PyArray_SimpleNewFromData(3, PyArray_DIMS(wet_array), PyArray_TYPE(wet_array), DoseArray.release());

		PyArray_ENABLEFLAGS((PyArrayObject*) return_dose, NPY_ARRAY_OWNDATA);

		return return_dose;

	} catch (std::bad_alloc &) {

		PyErr_SetString(PyExc_MemoryError, "Not enough host memory");

	} catch (std::runtime_error &e) {

		PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", e.what());

	}

	return NULL;

}


static PyObject * photon_dose(PyObject* self, PyObject* args) {

	PyObject *model_instance, *volume_instance, *cp_instance;
	int gpu_id;	

	// parse arguments
    if (!PyArg_ParseTuple(args, "OOOi", &model_instance, &volume_instance, &cp_instance, &gpu_id))
        return NULL;

	// check beam model properties 
	double mu_cal, 
		primary_source_distance, 
		scatter_source_distance, 
		primary_source_size, 
		scatter_source_size, 
		mlc_distance, 
		scatter_source_weight, 
		electron_attenuation, 
		electron_src_weight,
		electron_fitted_dmax,
		jaw_transmission,
		mlc_transmission;

	if (!pyobject_getfloat(model_instance, "mu_calibration", &mu_cal)
	 || !pyobject_getfloat(model_instance, "primary_source_distance", &primary_source_distance)
	 || !pyobject_getfloat(model_instance, "scatter_source_distance", &scatter_source_distance)
	 || !pyobject_getfloat(model_instance, "primary_source_size", &primary_source_size)
	 || !pyobject_getfloat(model_instance, "scatter_source_size", &scatter_source_size)
	 || !pyobject_getfloat(model_instance, "mlc_distance", &mlc_distance)
	 || !pyobject_getfloat(model_instance, "scatter_source_weight", &scatter_source_weight)
	 || !pyobject_getfloat(model_instance, "electron_attenuation", &electron_attenuation)
	 || !pyobject_getfloat(model_instance, "electron_source_weight", &electron_src_weight)
	 || !pyobject_getfloat(model_instance, "electron_fitted_dmax", &electron_fitted_dmax)
	 || !pyobject_getfloat(model_instance, "jaw_transmission", &jaw_transmission)
	 || !pyobject_getfloat(model_instance, "mlc_transmission", &mlc_transmission)) {
		return NULL;
	}

	bool has_xjaws, 
		has_yjaws;
	if (!pyobject_getbool(model_instance, "has_xjaws", &has_xjaws)
	 || !pyobject_getbool(model_instance, "has_yjaws", &has_yjaws)) {
		return NULL;
	}

	PyArrayObject *profile_radius, 
		*profile_intensities, 
		*profile_softening, 
		*spectrum_attenuation_coefficients, 
		*spectrum_primary_weights, 
		*spectrum_scatter_weights, 
		*kernel;
	if (!pyobject_getarray(model_instance, "profile_radius", 1, &profile_radius)
	 || !pyobject_getarray(model_instance, "profile_intensities", 1, &profile_intensities)
	 || !pyobject_getarray(model_instance, "profile_softening", 1, &profile_softening)
	 || !pyobject_getarray(model_instance, "spectrum_attenuation_coefficients", 1, &spectrum_attenuation_coefficients)
	 || !pyobject_getarray(model_instance, "spectrum_primary_weights", 1, &spectrum_primary_weights)
	 || !pyobject_getarray(model_instance, "spectrum_scatter_weights", 1, &spectrum_scatter_weights)
	 || !pyobject_getarray(model_instance, "kernel", 2, &kernel)) {
		return NULL;
	}

	// check volume data properties
	PyArrayObject *density_array, 
		*spacing_array, 
		*origin_array;
	if (!pyobject_getarray(volume_instance, "voxel_data", 3, &density_array)
	 || !pyobject_getarray(volume_instance, "spacing", 1, &spacing_array)
	 || !pyobject_getarray(volume_instance, "origin", 1, &origin_array)) {
		return NULL;
	}

	// check control point data properties
	PyArrayObject *iso_array, 
		*mlc_array;
	if (!pyobject_getarray(cp_instance, "iso", 1, &iso_array)
	 || !pyobject_getarray(cp_instance, "mlc", 2, &mlc_array)) {
		return NULL;
	}

	double mu, 
		ga, 
		ca, 
		ta;
	if (!pyobject_getfloat(cp_instance, "mu", &mu)
	 || !pyobject_getfloat(cp_instance, "ga", &ga)
	 || !pyobject_getfloat(cp_instance, "ca", &ca)
	 || !pyobject_getfloat(cp_instance, "ta", &ta)) {
		return NULL;
	}

	float * spacing = pyarray_as<float>(spacing_array);
	float * origin = pyarray_as<float>(origin_array);
	float * iso = pyarray_as<float>(iso_array);
	double voxel_sp = (double)spacing[0];

	size_t n_mlc_pairs = PyArray_DIM(mlc_array, 0);

	try {

		float adjusted_ga = fmodf(ga + 180.0f, 360.0f);

		size_t dims[3] = {
			(size_t)PyArray_DIMS(density_array)[0],
			(size_t)PyArray_DIMS(density_array)[1],
			(size_t)PyArray_DIMS(density_array)[2],
		};

		float adjusted_isocenter[3] = {
			iso[0] - origin[0],
			iso[1] - origin[1],
			iso[2] - origin[2]
		};

		// beam model object
		auto model = IMRTBeam::Model();
		model.n_profile_points = PyArray_DIM(profile_radius, 0);
		model.profile_radius = pyarray_as<float>(profile_radius);
		model.profile_intensities = pyarray_as<float>(profile_intensities);
		model.profile_softening = pyarray_as<float>(profile_softening);
		model.n_spectral_energies = PyArray_DIM(spectrum_attenuation_coefficients, 0);
		model.spectrum_attenuation_coefficients = pyarray_as<float>(spectrum_attenuation_coefficients);
		model.spectrum_primary_weights = pyarray_as<float>(spectrum_primary_weights);
		model.spectrum_scatter_weights = pyarray_as<float>(spectrum_scatter_weights);
		model.mu_cal = mu_cal;
		model.primary_src_dist = primary_source_distance;
		model.scatter_src_dist = scatter_source_distance;
		model.primary_src_size = primary_source_size;
		model.scatter_src_size = scatter_source_size;
		model.mlc_distance = mlc_distance;
		model.scatter_src_weight = scatter_source_weight;
		model.electron_attenuation = electron_attenuation;
		model.electron_src_weight = electron_src_weight;
		model.kernel = pyarray_as<float>(kernel);
		model.has_xjaws = has_xjaws;
		model.has_yjaws = has_yjaws;
		model.electron_fitted_dmax = electron_fitted_dmax;
		model.jaw_transmission = jaw_transmission;
		model.mlc_transmission = mlc_transmission;

		// dose object
		IMRTDose dose_obj = IMRTDose(dims, voxel_sp);
		HostPointer<float> WETArray(dose_obj.num_voxels);
		HostPointer<float> DoseArray(dose_obj.num_voxels);

		dose_obj.DensityArray = pyarray_as<float>(density_array);
		dose_obj.WETArray = WETArray.get();
		dose_obj.DoseArray = DoseArray.get();

		// MLC object_array
		HostPointer<MLCPair> MLCPairArray(n_mlc_pairs);
		make_mlc_array(mlc_array, MLCPairArray);

		// beam object
		IMRTBeam beam_obj = IMRTBeam(adjusted_isocenter, adjusted_ga, ta, ca, &model);
		beam_obj.n_mlc_pairs = n_mlc_pairs;
		beam_obj.mlc = MLCPairArray.get();
		beam_obj.mu = mu;

		// compute dose
    	photon_dose_cuda(gpu_id, &dose_obj, &beam_obj);

		PyObject *return_dose = PyArray_SimpleNewFromData(3, PyArray_DIMS(density_array), PyArray_TYPE(density_array), DoseArray.release());

		PyArray_ENABLEFLAGS((PyArrayObject*) return_dose, NPY_ARRAY_OWNDATA);

		return return_dose;

	} catch (std::bad_alloc &) {

		PyErr_SetString(PyExc_MemoryError, "Not enough host memory");

	} catch (std::runtime_error &e) {

		PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", e.what());

	}

	return NULL;

}


static PyMethodDef DoseMethods[] = {
	{
		"proton_raytrace_cuda",
		proton_raytrace,
		METH_VARARGS,
		"Compute WET array for proton dose calc."
	},
	{
		"proton_spot_cuda",
		proton_spot,
		METH_VARARGS,
		"Compute proton spot dose with PB using pre-calc'd WET array."
	},
	{
		"photon_dose_cuda",
		photon_dose,
		METH_VARARGS,
		"Compute proton spot dose with PB using pre-calc'd WET array."
	},
	{ 0 }
};


static struct PyModuleDef dosemodule = {
	PyModuleDef_HEAD_INIT,
	"dose_kernels",
	"Compute dose on the GPU.",
	-1,
	DoseMethods,
};


PyMODINIT_FUNC PyInit_dose_kernels(void) {
	import_array();
	return PyModule_Create(&dosemodule);
}

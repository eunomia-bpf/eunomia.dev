#ifndef CUPTI_ACTIVITY_COMPAT_H_
#define CUPTI_ACTIVITY_COMPAT_H_

#include <cupti.h>

// CUPTI 13.2 replaced CUpti_ActivityDevice5 with CUpti_ActivityDevice6.
#if CUPTI_API_VERSION >= 130200
typedef CUpti_ActivityDevice6 CuptiActivityDevice;
#else
typedef CUpti_ActivityDevice5 CuptiActivityDevice;
#endif

#endif // CUPTI_ACTIVITY_COMPAT_H_

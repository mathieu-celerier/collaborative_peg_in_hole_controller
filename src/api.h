#pragma once

#if defined _WIN32 || defined __CYGWIN__
#  define CollabPegInHoleController_DLLIMPORT __declspec(dllimport)
#  define CollabPegInHoleController_DLLEXPORT __declspec(dllexport)
#  define CollabPegInHoleController_DLLLOCAL
#else
// On Linux, for GCC >= 4, tag symbols using GCC extension.
#  if __GNUC__ >= 4
#    define CollabPegInHoleController_DLLIMPORT __attribute__((visibility("default")))
#    define CollabPegInHoleController_DLLEXPORT __attribute__((visibility("default")))
#    define CollabPegInHoleController_DLLLOCAL __attribute__((visibility("hidden")))
#  else
// Otherwise (GCC < 4 or another compiler is used), export everything.
#    define CollabPegInHoleController_DLLIMPORT
#    define CollabPegInHoleController_DLLEXPORT
#    define CollabPegInHoleController_DLLLOCAL
#  endif // __GNUC__ >= 4
#endif // defined _WIN32 || defined __CYGWIN__

#ifdef CollabPegInHoleController_STATIC
// If one is using the library statically, get rid of
// extra information.
#  define CollabPegInHoleController_DLLAPI
#  define CollabPegInHoleController_LOCAL
#else
// Depending on whether one is building or using the
// library define DLLAPI to import or export.
#  ifdef CollabPegInHoleController_EXPORTS
#    define CollabPegInHoleController_DLLAPI CollabPegInHoleController_DLLEXPORT
#  else
#    define CollabPegInHoleController_DLLAPI CollabPegInHoleController_DLLIMPORT
#  endif // CollabPegInHoleController_EXPORTS
#  define CollabPegInHoleController_LOCAL CollabPegInHoleController_DLLLOCAL
#endif // CollabPegInHoleController_STATIC

#include "streaming/onevpl/accelerators/surface/surface_pool.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "logger.hpp"

#ifdef HAVE_ONEVPL

namespace cv {
namespace gapi {
namespace wip {

void CachedPool::reserve(size_t size) {
    surfaces.reserve(size);
}

size_t CachedPool::size() const {
    return surfaces.size();
}

void CachedPool::clear() {
    surfaces.clear();
    next_free_it = surfaces.begin();
    cache.clear();
}

void CachedPool::push_back(surface_ptr_t &&surf) {
    cache.insert(std::make_pair(surf->get_handle(), surf));
    surfaces.push_back(std::move(surf));
    next_free_it = surfaces.begin();
}

CachedPool::surface_ptr_t CachedPool::find_free() {
    auto it =
        std::find_if(next_free_it, surfaces.end(),
                     [](const surface_ptr_t& val) {
            GAPI_DbgAssert(val && "Pool contains empty surface");
            return !val->get_locks_count();
        });

    // Limitation realloc pool might be a future extension
    if (it == surfaces.end()) {
        it = std::find_if(surfaces.begin(), next_free_it,
                          [](const surface_ptr_t& val) {
                GAPI_DbgAssert(val && "Pool contains empty surface");
                return !val->get_locks_count();
            });
        if (it == next_free_it) {
            std::stringstream ss;
            ss << "cannot get free surface from pool, size: " << surfaces.size();
            const std::string& str = ss.str();
            GAPI_LOG_WARNING(nullptr, str);
            throw std::runtime_error(std::string(__FUNCTION__) + " - " + str);
        }
    }

    next_free_it = it;
    ++next_free_it;

    return *it;
}

CachedPool::surface_ptr_t CachedPool::find_by_handle(mfxFrameSurface1* handle) {
    auto it = cache.find(handle);
    GAPI_Assert(it != cache.end() && "Cannot find cached surface from pool. Data corruption is possible");
    return it->second;
}
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL

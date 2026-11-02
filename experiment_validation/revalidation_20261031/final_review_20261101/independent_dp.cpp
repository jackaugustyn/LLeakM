#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <vector>

// Independent textbook dynamic programs, distinct from the bit-vector metric
// implementation used by the experiment verifier. Inputs are Unicode code
// points for ED and word IDs for LCS; neither function uses model libraries.
extern "C" uint32_t edit_distance(const uint32_t* a, size_t n,
                                   const uint32_t* b, size_t m) {
    std::vector<uint32_t> previous(m+1), current(m+1);
    for (size_t j=0;j<=m;++j) previous[j]=j;
    for (size_t i=1;i<=n;++i) {
        current[0]=i;
        for (size_t j=1;j<=m;++j)
            current[j]=std::min({previous[j]+1,current[j-1]+1,
                                previous[j-1]+uint32_t(a[i-1]!=b[j-1])});
        previous.swap(current);
    }
    return previous[m];
}

extern "C" uint32_t longest_common_subsequence(const uint32_t* a, size_t n,
                                                const uint32_t* b, size_t m) {
    std::vector<uint32_t> previous(m+1), current(m+1);
    for (size_t i=1;i<=n;++i) {
        current[0]=0;
        for (size_t j=1;j<=m;++j)
            current[j]=a[i-1]==b[j-1] ? previous[j-1]+1
                                      : std::max(previous[j],current[j-1]);
        previous.swap(current);
    }
    return previous[m];
}

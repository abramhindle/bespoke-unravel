//
//  Copyright(C) 2012 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#include "codec/lz4.hpp"
#include "codec/lz4.h"
#include "codec/codec_impl.hpp"

#include <iostream>

#include "utils/bithack.hpp"

namespace codec
{
  namespace detail
  {
    const lz4_param::size_type lz4_param::chunk_size = 1024 * 1024;
    const lz4_param::size_type lz4_param::bound_size = LZ4_compressBound(1024 * 1024) + 4;
  };

  lz4_compressor_impl::lz4_compressor_impl()
    : buffer(new byte_type[chunk_size]),
      pos(0),
      buffer_compressed(new byte_type[bound_size]),
      size_compressed(0),
      pos_compressed(0)
  { }
  
  lz4_compressor_impl::lz4_compressor_impl(const lz4_compressor_impl&)
    : buffer(new byte_type[chunk_size]),
      pos(0),
      buffer_compressed(new byte_type[bound_size]),
      size_compressed(0),
      pos_compressed(0)
  { }
  
  bool lz4_compressor_impl::filter(const char*& src_begin,
				   const char* src_end,
				   char*& dest_begin,
				   char* dest_end,
				   bool flush)
  {
    const size_type src_copied = utils::bithack::min(size_type(src_end - src_begin), chunk_size - pos);
      
    // copy into buffer
    if (src_copied) {
      std::copy(src_begin, src_begin + src_copied, buffer + pos);
      src_begin += src_copied;
      pos += src_copied;
    }
      
    // perform compression, if all the compressed buffer is dumped, pos == chunk or flush
    if (pos_compressed == size_compressed && (pos == chunk_size || (pos && flush))) {
      size_compressed = LZ4_compress(buffer, buffer_compressed + 4, pos);
      
      impl::write_size(size_compressed, buffer_compressed);
      
      size_compressed += 4;
      pos_compressed = 0;
      pos = 0;
    }
    
    // perform copy into dest
    const size_type dest_copied = utils::bithack::min(size_type(dest_end - dest_begin), size_compressed - pos_compressed);
    
    if (dest_copied) {
      std::copy(buffer_compressed + pos_compressed, buffer_compressed + pos_compressed + dest_copied, dest_begin);
      dest_begin += dest_copied;
      pos_compressed += dest_copied;
    }
    
    return (size_compressed != pos_compressed) || pos;
  }


  lz4_decompressor_impl::lz4_decompressor_impl()
    : buffer_compressed(new byte_type[bound_size]),
      size_compressed(0),
      pos_compressed(0),
      buffer(new byte_type[chunk_size]),
      size(0),
      pos(0)
  { }
  
  lz4_decompressor_impl::lz4_decompressor_impl(const lz4_decompressor_impl&)
    : buffer_compressed(new byte_type[bound_size]),
      size_compressed(0),
      pos_compressed(0),
      buffer(new byte_type[chunk_size]),
      size(0),
      pos(0)
  { }

  /*
  int lz4_decompressor_impl::check_end(const char* src_begin, const char* dest_begin) 
  { 
	  bz_stream* s = static_cast<bz_stream*>(stream_); 
	  if( src_begin == s->next_in && 
			  s->avail_in == 0 && 
			  dest_begin == s->next_out) { 
		  return bzip2::unexpected_eof; 
	  } else { 
		  return bzip2::ok; 
	  } 
  }*/
  
  bool lz4_decompressor_impl::filter(const char*& src_begin,
				     const char* src_end,
				     char*& dest_begin,
				     char* dest_end,
				     bool flush)
  {
    // copy into buffer as much as possible
    
   
    const size_type src_copied = utils::bithack::min(size_type(src_end - src_begin),
						     utils::bithack::branch(size_compressed,
									    size_compressed + 4,
									    bound_size)
						     - pos_compressed);
 
    if (src_copied) {
      std::copy(src_begin, src_begin + src_copied, buffer_compressed + pos_compressed);
      src_begin += src_copied;
      pos_compressed += src_copied;
    }

    // assig size-compressed if possible
    if ((! size_compressed) && pos_compressed >= 4)
      size_compressed = impl::read_size(buffer_compressed);

    // perform actual uncompression...
    if (pos == size && size_compressed && pos_compressed >= size_compressed + 4) {
      size = LZ4_uncompress_unknownOutputSize(buffer_compressed + 4, buffer, size_compressed, chunk_size);
      pos = 0;
      
      // copy with potential overlap...
      std::copy(buffer_compressed + size_compressed + 4, buffer_compressed + pos_compressed, buffer_compressed);
      pos_compressed -= size_compressed + 4;
      size_compressed = 0;
      
      // assig size-compressed if possible
      if ((! size_compressed) && pos_compressed >= 4)
	size_compressed = impl::read_size(buffer_compressed);
    }
   

    // dump into dest
    const size_type dest_copied = utils::bithack::min(size_type(dest_end - dest_begin), size - pos);
   
    if (dest_copied) {
      std::copy(buffer + pos, buffer + pos + dest_copied, dest_begin);
      dest_begin += dest_copied;
      pos += dest_copied;
    }

    // don't know how to fix that 
    // if(result == bzip2::ok && flush) result = check_end(src_begin, dest_begin);

    return (pos != size) || pos_compressed;
  }

}


///  Simple log system baced on the Nasa Vision Workbench log system.
/// This is a simple work around until boost.log is official and included in major linux distributions (around 2012 ?)
/// Did not find any "good enough" logging library http://stackoverflow.com/questions/696321/best-logging-framework-for-native-c
///
/// Aside from the standard basic ostream functionality, this set of
/// classes provides:
///
/// - Buffering of the log messages on a per-thread messages so that
///   messages form different threads are nicelely interleaved in the
///   log.
///
/// Some notes on the behavior of the log.
///
/// - A new line in the logfile starts every time a newline character
///   appears at the end of a string of characters, or when you
///   exlicitly add std::flush() to the stream of operators.

#ifndef LOG_HEADER_INCLUDED
#define LOG_HEADER_INCLUDED

// Boost Headers
#include <boost/algorithm/string.hpp>

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

// STD Headers
#include <string>
#include <list>
#include <vector>
#include <map>
#include <iostream>

// STD Headers for defininog our own ostream subclass
#include <streambuf>
#include <ostream>
#include <fstream>

// For stringify
#include <sstream>

namespace logging {


  // ----------------------------------------------------------------
  // Debugging output types and functions
  // ----------------------------------------------------------------

  // Lower number -> higher priority
  enum MessageLevel {
    ErrorMessage = 0,
    WarningMessage = 10,
    InfoMessage = 20,
    DebugMessage = 30,
    VerboseDebugMessage = 40,
    EveryMessage = 100
  };

  /// \cond INTERNAL

  // ----------------------------------------------------------------
  //                     Utility Streams
  // ----------------------------------------------------------------
  //
  // These classes provide a basic NULL ostream and an ostream that
  // can take data and re-stream it to multiple sub-streams.

  // Null Output Stream
  template<typename CharT, typename traits = std::char_traits<CharT> >
  class NullOutputBuf : public std::basic_streambuf<CharT, traits> {
    typedef typename std::basic_streambuf<CharT, traits>::int_type int_type;
    virtual int_type overflow(int_type c) { return traits::not_eof(c); }
    virtual std::streamsize xsputn(const CharT* /*sequence*/, std::streamsize num) { return num; }
  };

  template<typename CharT, typename traits>
  class NullOutputStreamInit {
    NullOutputBuf<CharT, traits> m_buf;
  public:
    NullOutputBuf<CharT, traits>* buf() { return &m_buf; }
  };

  template<typename CharT, typename traits = std::char_traits<CharT> >
  class NullOutputStream : private virtual NullOutputStreamInit<CharT, traits>,
                           public std::basic_ostream<CharT, traits> {
  public:
    NullOutputStream() : NullOutputStreamInit<CharT, traits>(),
                         std::basic_ostream<CharT,traits>(NullOutputStreamInit<CharT, traits>::buf()) {
        // nothing to do here
        return;
    }
  };

  // Multi output stream
  template<typename CharT, typename traits = std::char_traits<CharT> >
  class MultiOutputBuf : public std::basic_streambuf<CharT, traits> {
    typedef typename std::basic_streambuf<CharT, traits>::int_type int_type;
    typedef std::vector<std::basic_ostream<CharT, traits>* > stream_container;
    typedef typename stream_container::iterator stream_iterator;
    stream_container m_streams;
    boost::mutex m_mutex;

  protected:
    virtual std::streamsize xsputn(const CharT* sequence, std::streamsize num) {
      boost::mutex::scoped_lock lock(m_mutex);
      stream_iterator current = m_streams.begin();
      stream_iterator end = m_streams.end();
      for(; current != end; ++current)
        (*current)->write(sequence, num);
      return num;
    }

    virtual int_type overflow(int_type c) {
      boost::mutex::scoped_lock lock(m_mutex);
      stream_iterator current = m_streams.begin();
      stream_iterator end = m_streams.end();

      for(; current != end; ++current)
        (*current)->put(c);
      return c;
    }

    // The sync function simply propogates the sync() request to the
    // child ostreams.
    virtual int sync() {
      boost::mutex::scoped_lock lock(m_mutex);
      stream_iterator current = m_streams.begin();
      stream_iterator end = m_streams.end();

      for(; current != end; ++current)
        (*current)->rdbuf()->pubsync();
      return 0;
    }

  public:
    void add(std::basic_ostream<CharT, traits>& stream) {
      boost::mutex::scoped_lock lock(m_mutex);
      m_streams.push_back(&stream);
    }
    void remove(std::basic_ostream<CharT, traits>& stream) {
      boost::mutex::scoped_lock lock(m_mutex);
      stream_iterator pos = std::find(m_streams.begin(),m_streams.end(), &stream);
      if(pos != m_streams.end())
        m_streams.erase(pos);
    }
    void clear() {
      boost::mutex::scoped_lock lock(m_mutex);
      m_streams.clear();
    }
  };

  template<typename CharT, typename traits>
  class MultiOutputStreamInit {
    MultiOutputBuf<CharT, traits> m_buf;
  public:
    MultiOutputBuf<CharT, traits>* buf() { return &m_buf; }
  };

  template<typename CharT, typename traits = std::char_traits<CharT> >
  class MultiOutputStream : private MultiOutputStreamInit<CharT, traits>,
                            public std::basic_ostream<CharT, traits> {
  public:
    MultiOutputStream() : MultiOutputStreamInit<CharT, traits>(),
                            std::basic_ostream<CharT, traits>(MultiOutputStreamInit<CharT, traits>::buf()) {}
    void add(std::basic_ostream<CharT, traits>& str) { MultiOutputStreamInit<CharT, traits>::buf()->add(str); }
    void remove(std::basic_ostream<CharT, traits>& str) { MultiOutputStreamInit<CharT, traits>::buf()->remove(str); }
    void clear() { MultiOutputStreamInit<CharT, traits>::buf()->clear(); }
  };

  // Some handy typedefs
  //
  // These are made to be lower case names to jive with the C++ std
  // library naming conversion for streams (i.e. std::cin, std::cout,
  // etc.)
  typedef NullOutputStream<char> null_ostream;
  typedef MultiOutputStream<char> multi_ostream;


  // In order to create our own C++ streams compatible ostream object,
  // we must first define a subclass of basic_streambuf<>, which
  // handles stream output on a character by character basis.  This is
  // not the most elegent block of code, but this seems to be the
  // "approved" method for defining custom behaviour in a subclass of
  // basic_ostream<>.
  template<class CharT, class traits = std::char_traits<CharT> >
  class PerThreadBufferedStreamBuf : public std::basic_streambuf<CharT, traits> {

    typedef typename std::basic_streambuf<CharT, traits>::int_type int_type;

    // Characters are buffered is vectors until a newline appears at
    // the end of a line of input or flush() is called.  These vectors
    // are indexed in a std::map by thread id.
    //
    // TODO: This map could grow quite large if a program spawns (and
    // logs to) many, many threads.  We should think carefully about
    // cleaning up this map structure from time to time.
    typedef std::vector<CharT> buffer_type;
    typedef std::map<boost::thread::id, buffer_type> lookup_table_type;
    lookup_table_type m_buffers;

    std::basic_streambuf<CharT, traits>* m_out;
    boost::mutex m_mutex;

    // This method is called when a single character is fed to the
    // streambuf.  In practice, characters are fed in batches using
    // xputn() below.
    virtual int_type overflow(int_type c) {
      {
        boost::mutex::scoped_lock lock(m_mutex);
        if(not traits::eq_int_type(c, traits::eof())) {
          m_buffers[ boost::this_thread::get_id() ].push_back(static_cast<CharT>(c));
        }
      }

      // If the last character is a newline or cairrage return, then
      // we force a call to sync().
      if ( c == '\n' or c == '\r' )
        sync();
      return traits::not_eof(c);
    }

    virtual std::streamsize xsputn(const CharT* s, std::streamsize num) {
      buffer_type& buffer = m_buffers[ boost::this_thread::get_id() ];
      {
        boost::mutex::scoped_lock lock(m_mutex);
        std::copy(s, s + num, std::back_inserter<buffer_type>( buffer ));
      }

      // This is a bit of a hack that forces a sync whenever the
      // character string *ends* with a newline, thereby flushing the
      // buffer and printing a line to the log file.
      if ( buffer.size() > 0 ) {
        const size_t last_char_position = buffer.size()-1;

        if ( buffer[last_char_position] == '\n' or
             buffer[last_char_position] == '\r' )
        {
          sync();
      }
      }
      return num;
    }

    virtual int sync() {
      boost::mutex::scoped_lock lock(m_mutex);
      if(not m_buffers[ boost::this_thread::get_id() ].empty() and m_out ) {
        m_out->sputn(&m_buffers[ boost::this_thread::get_id() ][0], static_cast<std::streamsize>(m_buffers[ boost::this_thread::get_id() ].size()));
        m_out->pubsync();
        m_buffers[ boost::this_thread::get_id() ].clear();
      }
      return 0;
    }

  public:
    PerThreadBufferedStreamBuf() : m_buffers(), m_out(NULL) {}
    ~PerThreadBufferedStreamBuf() { sync(); }

    void init(std::basic_streambuf<CharT,traits>* out) { m_out = out; }
  };

  // The order with which the base classes are initialized in
  // PerThreadBufferedStream is not fully defined unless we inherit from this as a
  // pure virtual base class.  This gives us the extra wiggle room we
  // need to connect two properly initialized PerThreadBufferedStreamBuf and
  // PerThreadBufferedStream objects.
  template<class CharT, class traits = std::char_traits<CharT> >
  class PerThreadBufferedStreamBufInit {
    PerThreadBufferedStreamBuf<CharT, traits> m_buf;
  public:
    PerThreadBufferedStreamBuf<CharT, traits>* buf() {
      return &m_buf;
    }
  };

  // Aside from some tricky initialization semantics, this subclass of
  // basic_ostream is actually fairly simple.  It passes along
  // characters to the PerThreadBufferedStreamBuf, which does the
  // actual interesting stuff.
  //
  // You can use the ctor or the set_stream method to pass in any C++
  // ostream (e.g. std::cout or a std::ofstream) to be the ultimate
  // recipient of the characters that are fed through this stream,
  // which acts as an intermediary, queuing characters on a per thread
  // basis and ensuring thread safety.
  template<class CharT, class traits = std::char_traits<CharT> >
  class PerThreadBufferedStream : private virtual PerThreadBufferedStreamBufInit<CharT, traits>,
                                  public std::basic_ostream<CharT, traits> {
  public:
    // No stream specified.  Will swallow characters until one is set
    // using set_stream().
    PerThreadBufferedStream() : PerThreadBufferedStreamBufInit<CharT,traits>(),
                                std::basic_ostream<CharT, traits>(PerThreadBufferedStreamBufInit<CharT,traits>::buf()) {}


    PerThreadBufferedStream(std::basic_ostream<CharT, traits>& out) : PerThreadBufferedStreamBufInit<CharT,traits>(),
                                                                      std::basic_ostream<CharT, traits>(PerThreadBufferedStreamBufInit<CharT,traits>::buf()) {
      PerThreadBufferedStreamBufInit<CharT,traits>::buf()->init(out.rdbuf());
    }

    void set_stream(std::basic_ostream<CharT, traits>& out) {
      PerThreadBufferedStreamBufInit<CharT,traits>::buf()->init(out.rdbuf());
    }

  };

  /// \endcond

  template<typename T>
  inline std::string stringify(const T& x)
  {
    std::ostringstream o;
    if (not (o << x))
    {
      return std::string("[failed to stringify ") + typeid(x).name() + "]";
  }
    return o.str();
  }

  class LogRuleSet {
    // The ruleset determines what log messages are sent to the system log file
    typedef std::pair<int, std::string> rule_type;
    typedef std::list<rule_type> rules_type;
    rules_type m_rules;
    boost::mutex m_mutex;

    // Help functions
    inline bool has_leading_wildcard( std::string const& exp ) {
      size_t index = exp.rfind("*");
      if ( index == std::string::npos )
        return false;
      return size_t(exp.size())-1 > index;
    }

    inline std::string after_wildcard( std::string const& exp ) {
      int index = exp.rfind("*");
      if ( index != -1 ) {
        index++;
        return exp.substr(index,exp.size()-index);
      }
      return "";
    }

  public:

    // Ensure Copyable semantics
    LogRuleSet( LogRuleSet const& copy_log) {
      m_rules = copy_log.m_rules;
    }

    LogRuleSet& operator=( LogRuleSet const& copy_log) {
      m_rules = copy_log.m_rules;
      return *this;
    }


    // by default, the LogRuleSet is set up to pass "console" messages
    // at level vw::WarningMessage or higher priority.
    LogRuleSet() {
      m_rules.push_back(rule_type(logging::InfoMessage, "console"));
    }

    virtual ~LogRuleSet() {}

    void add_rule(int log_level, std::string log_namespace) {
      boost::mutex::scoped_lock lock(m_mutex);
      m_rules.push_front(rule_type(log_level, boost::to_lower_copy(log_namespace)));
    }

    void clear() {
      boost::mutex::scoped_lock lock(m_mutex);
      m_rules.clear();
    }

    // You can overload this method from a subclass to change the
    // behavior of the LogRuleSet.
    virtual bool operator() (int log_level, std::string log_namespace) {
      boost::mutex::scoped_lock lock(m_mutex);

      std::string lower_namespace = boost::to_lower_copy(log_namespace);

      for (rules_type::iterator it = m_rules.begin(); it != m_rules.end(); ++it) {

        // Pass through rule for complete wildcard
        if ( (*it).second == "*" and
             ( (*it).first == logging::EveryMessage or
               log_level <= (*it).first ) )
          {
          return true;
      }

        // For explicit matching on namespace
        if ( (*it).second == lower_namespace ) {
          if ( log_level <= (*it).first )
            {
            return true;
        }
          else
          {
            return false;
        }
        }

        // Evaluation of half wild card
        if ( has_leading_wildcard( (*it).second )  and
             boost::iends_with(lower_namespace,after_wildcard((*it).second)) ) {
          if ( log_level <= (*it).first )
            {
            return true;
        }
          else
          {
            return false;
        }
        }
      }

      // Progress bars get a free ride at InfoMessage level unless a
      // rule above modifies that.
      if ( boost::iends_with(lower_namespace,".progress") and
           log_level == logging::InfoMessage )
      {
        return true;
    }

      // We reach this line if all of the rules have failed, in
      // which case we return a NULL stream, which will result in
      // nothing being logged.
      return false;
    }
  };


  // -------------------------------------------------------
  //                         LogInstance
  // -------------------------------------------------------
  //
  class LogInstance {

  protected:
    PerThreadBufferedStream<char> m_log_stream;
    std::ostream *m_log_ostream_ptr;
    bool m_prepend_infostamp;
    LogRuleSet m_rule_set;

  private:
    // Ensure non-copyable semantics
    LogInstance( LogInstance const& );
    LogInstance& operator=( LogInstance const& );

  public:

    // Initialize a log from a filename.  A new internal ofstream is
    // created to stream log messages to disk.
    LogInstance(std::string log_filename, bool prepend_infostamp = true);

    // Initialize a log using an already open stream.  Warning: The
    // log stores the stream by reference, so you MUST delete the log
    // object _before_ closing and de-allocating the stream.
    LogInstance(std::ostream& log_ostream, bool prepend_infostamp = true);

    ~LogInstance() {
      m_log_stream.set_stream(std::cout);
      if (m_log_ostream_ptr)
        delete static_cast<std::ofstream*>(m_log_ostream_ptr);
    }

    /// This method return an ostream that you can write a log message
    /// to if the rule_set matches the log level and namespace
    /// provided.  Otherwise, a null ostream is returned.
    virtual std::ostream& operator() (const int log_level, const std::string log_namespace="console");

    /// Access the rule set for this log object.
    LogRuleSet& rule_set() { return m_rule_set; }
  };


  // -------------------------------------------------------
  //                         Log
  // -------------------------------------------------------

  /// The system log class manages logging to the console and to files
  /// on disk.  It supports multiple open log streams, each with their
  /// own LogRuleSet.
  ///
  /// Important Note: You should access the system log using the
  /// Log::system_log() static method, which access a singleton
  /// instance of the system log class.  You should not need to create
  /// a log object yourself.
  class Log {

    // Pointers to various log instances that are currently being
    // managed by the system log.
    std::vector<boost::shared_ptr<LogInstance> > m_logs;
    boost::shared_ptr<LogInstance> m_console_log;

    // Member variables
    boost::mutex m_system_log_mutex;
    boost::mutex m_multi_ostreams_mutex;

    // The multi_ostream creates a single stream that delegates to its
    // child streams. We store one multi_ostream per thread, since
    // each thread will have a different set of output streams it is
    // currently accessing.
    //
    // TODO: Rather than manage this as a simple map (which will grow
    // quite large and hurt performance if there are many threads), we
    // should really use some sort of cache of shared_ptr's to
    // ostreams here.  The tricky thing is that once we return the
    // ostream, we don't know how long the thread will use it before
    // it can be safely de-allocated.
    std::map<boost::thread::id, boost::shared_ptr<multi_ostream> > m_multi_ostreams;

    // Ensure non-copyable semantics
    Log( Log const& );
    Log& operator=( Log const& );

  public:

    /// You should probably not create an instance of Log on your own
    /// using this constructor.  Instead, you can access a global
    /// instance of the log class using the static Log::system_log()
    /// method below.
    Log() : m_console_log(new LogInstance(std::cout, false)) { }

    /// The call operator returns a subclass of the basic_ostream
    /// object, which is suitable for use with the C++ << operator.
    /// The returned stream object proxy's for the various log streams
    /// being managed by the system log that match the log_level and
    /// log_namespace.
    std::ostream& operator() (int log_level, std::string log_namespace="console");

    /// Add a stream to the Log manager.  You may optionally specify a
    /// LogRuleSet.
    void add(std::ostream &stream, LogRuleSet rule_set = LogRuleSet(), const bool prepend_infostamp = true) {
      boost::mutex::scoped_lock lock(m_system_log_mutex);
      m_logs.push_back( boost::shared_ptr<LogInstance>(new LogInstance(stream, prepend_infostamp)) );
      m_logs.back()->rule_set() = rule_set;
      return;
    }

    // Add an already existing LogInstance to the system log manager.
    void add(boost::shared_ptr<LogInstance> log) {
      boost::mutex::scoped_lock lock(m_system_log_mutex);
      m_logs.push_back( log );
      return;
    }

    /// Reset the System Log; closing all of the currently open Log
    /// streams.
    void clear() {
      boost::mutex::scoped_lock lock(m_system_log_mutex);
      m_logs.clear();
      return;
    }

    /// @returns a reference to the console LogInstance.
    LogInstance& console_log() {
      boost::mutex::scoped_lock lock(m_system_log_mutex);
      return *m_console_log;
    }

    void set_console_log(boost::shared_ptr<LogInstance> &log_p)
    {
        m_console_log = log_p;
        return;
    }

    /// Set the output stream and LogRuleSet for the console log
    /// instance.  This can be used to redirect the console output to
    /// a file, for example.
    void set_console_stream(std::ostream& stream, LogRuleSet rule_set = LogRuleSet(), bool prepend_infostamp = true) {
      boost::mutex::scoped_lock lock(m_system_log_mutex);
      m_console_log = boost::shared_ptr<LogInstance>(new LogInstance(stream, prepend_infostamp) );
      m_console_log->rule_set() = rule_set;
      return;
    }
  };

  /// Static method to access the singleton instance of the system
  /// log.  You should *always* use this method if you want to access
  /// the system log, where all log messages go.
  /// For example:
  ///
  ///     get_log().console_log() << "Some text\n";
  ///
  Log& get_log();

  /// The vision workbench logging operator.  Use this to generate a
  /// message in the system log using the given log_level and
  /// log_namespace.
  std::ostream& log( int log_level = logging::InfoMessage,
                        std::string log_namespace = "console" );


  /// Helpers used for extension classes
  /// {
  std::string current_posix_time_string();
  extern logging::null_ostream g_null_ostream;
  /// }

} // end of namespace logging

#endif // LOG_HEADER_INCLUDED

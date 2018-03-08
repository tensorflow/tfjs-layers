module.exports = function (config) {
  config.set({
    frameworks: ['jasmine', 'karma-typescript'],
    files: [{ pattern: 'src/**/*.ts' }],
    preprocessors: {
      '**/*.ts': ['karma-typescript'],  // *.tsx for React Jsx
    },
    karmaTypescriptConfig: { tsconfig: 'tsconfig.json' },
    reporters: ['progress', 'karma-typescript'],
    reportSlowerThan: 500,
    browsers: ['Chrome', 'Firefox'],
    client: {
      args: ['--grep', config.grep || '']
    }
  });
};
